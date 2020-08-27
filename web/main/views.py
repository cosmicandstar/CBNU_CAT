import csv
import io
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import sklearn
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist, jaccard
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render, get_object_or_404, get_list_or_404
from django.urls import reverse
from django.core.paginator import Paginator


from .models import *


# Create your views here.
def index(request):
    return render(request, 'main/index.html')


def login(request):
    try:
        request.session['userid']
        return HttpResponseRedirect(reverse('myinfo'))
    except:
        return render(request, 'main/login.html')


def sign_in(request):
    user_id = request.POST['user_id']
    user_pw = request.POST['user_pw']
    try:
        user = User.objects.get(userid=user_id, password=user_pw)
        request.session['userid'] = user_id
        return HttpResponseRedirect(reverse('myinfo'))
    except User.DoesNotExist:
        return HttpResponse('실패')


def sign_up(request):
    return render(request, 'main/signup.html')


def join(request):
    user_id = request.POST['id']
    user_pw = request.POST['pw']
    user_name = request.POST['name']
    user_number = request.POST['student_number']
    try:
        user = User.objects.get(userid=user_id)
        return HttpResponse('이미 존재하는 아이디')
    except User.DoesNotExist:
        user = User(userid=user_id, password=user_pw, username=user_name, user_number=user_number, mbti='None')
        user.save()

        keywords = SubjectKeyword.objects.all()
        for keyword in keywords:
            UserKeyword.objects.update_or_create(
                user_id=user.id,
                keyword_id=keyword.keyword_id,
                keyword=keyword.keyword,
                flag=0
            )
        request.session['userid'] = user_id
        return HttpResponseRedirect(reverse('myinfo'))

    return HttpResponseRedirect(reverse('signup'))


def myinfo(request):
    try:
        user = User.objects.get(userid=request.session['userid'])
        user_keyword = []
        try:
            keywords = SubjectKeyword.objects.all()
            try:
                user_keyword = get_list_or_404(UserKeyword, user_id=user.id)
            except:
                pass
            if len(keywords) > len(user_keyword):
                for keyword in keywords:
                    try:
                        UserKeyword.objects.get(user_id=user.id, keyword_id=keyword.keyword_id)
                    except:
                        UserKeyword.objects.update_or_create(
                            user_id=user.id,
                            keyword_id=keyword.keyword_id,
                            keyword=keyword.keyword,
                            flag=0
                        )
                user_keyword = get_list_or_404(UserKeyword, user_id=user.id)
        except:
            pass
        return render(request, 'main/myinfo.html', {'user': get_object_or_404(User, userid=request.session['userid']),
                                                    'user_keywords': user_keyword})
    except KeyError:
        return HttpResponseRedirect(reverse('index'))


def mbti(request):
    try:
        value = request.POST['hidden_mymbti']
        user = User.objects.get(userid=request.session['userid'])
        user.mbti = value
        user.save()
        return HttpResponseRedirect(reverse('myinfo'))
    except KeyError:
        return HttpResponseRedirect(reverse('myinfo'))


def keyword(request, id):
    try:
        user = User.objects.get(userid=request.session['userid'])
        user_keyword = UserKeyword.objects.get(user_id=user.id, keyword_id=id)
        if user_keyword.flag == 0:
            user_keyword.flag = 1
        else:
            user_keyword.flag = 0
        user_keyword.save()
        return HttpResponseRedirect(reverse('myinfo'))
    except KeyError:
        return HttpResponseRedirect(reverse('myinfo'))


def pre_lec(data):
    data_wide = data.pivot_table(index=["id"],
                                 columns='keyword',
                                 values='value')
    data_wide = data_wide.fillna(0)

    return data_wide


def keyword_sort(user_keyword, lecture_bin):

    user_keyword = user_keyword.set_index("keyword")
    user_keyword_list = list(user_keyword.iloc[:, 0])
    check = lecture_bin * user_keyword_list
    result_list = check.sum(axis=1).sort_values(axis=0, ascending=False).index

    return result_list


def total_sim(like_data, wish_data, score_data, lecture_data, alpha):
    # like_in = pd.read_csv(file_path + "ratings_good.csv")
    # wish_in = pd.read_csv(file_path + "ratings_wish.csv")
    # score_in = pd.read_csv(file_path + "ratings_score.csv")
    # lecture_in = pd.read_csv(file_path + "lecture_bin.csv", encoding='cp949')

    like_in = pd.DataFrame(like_data, columns=["id", "lecture", "rating"])
    score_in = pd.DataFrame(score_data, columns=["id", "lecture", "rating"])
    lecture_in = pd.DataFrame(lecture_data, columns=["id", "keyword", "value"])

    # user_keyword = pd.DataFrame(a, columns=["keyword","key"])

    lecture_wide = pre_lec(lecture_in)
    lecture_list = lecture_wide.sort_values(axis=0, ascending=True, by="id").index
    # 혹시 몰라서 정렬함.


    like_wide = pre(like_in)
    score_wide = pre(score_in)

    like_sim = sim(like_wide, lecture_list)
    score_sim = sim(score_wide, lecture_list)
    print(score_sim)
    lecture_sim = sim_lecture(lecture_wide, lecture_list)

    ts = like_sim * alpha[0] + score_sim * alpha[1] + lecture_sim * alpha[3]

    if len(wish_data) > 0:
        wish_in = pd.DataFrame(wish_data, columns=["id", "lecture"])
        wish_in["rating"] = 1
        wish_wide = pre(wish_in)
        wish_sim = sim_bin(wish_wide, lecture_list)
        ts += + wish_sim * alpha[2]

    """like_sim.to_csv("like_sim.csv")
    wish_sim.to_csv("wish_sim.csv")
    lecture_sim.to_csv("lecture_sim.csv")
    score_sim.to_csv("score_sim.csv")"""

    return ts, lecture_list


def pre(data):
    data_wide = data.pivot_table(index=["id"],
                                 columns='lecture',
                                 values='rating')
    data_wide = data_wide.fillna(0)

    return data_wide


def pre_lec(data):
    data_wide = data.pivot_table(index=["id"],
                                 columns='keyword',
                                 values='value')
    data_wide = data_wide.fillna(0)

    return data_wide


def sim(data_wide, lecture_list):
    # data_sim = sklearn.metrics.pairwise.cosine_similarity(data_wide.T, dense_output=True)
    # data_sim = pd.DataFrame(data_sim,index=list(data_wide),columns=list(data_wide))
    # data_sim = data_sim - np.eye(len(data_sim))

    data_sim = sklearn.metrics.pairwise_distances(data_wide.T, metric='cosine')
    data_sim = pd.DataFrame(data_sim, index=list(data_wide), columns=list(data_wide))

    data_sim = 1 - data_sim - np.eye(len(data_sim))
    data_sim = pd.DataFrame(data_sim, index=lecture_list, columns=lecture_list)
    data_sim = data_sim.fillna(0)

    # 데이터의 수가 희소하므로 표준화는 배제한다.
    # 표준화가 필요하다면 0을 제외하고 진행할 것. 4/9 와 9 가 있을 때, 4/9의 9가 더 작아질 위험이 있음.

    # euclidean_distance & 정규화
    """data_sim = euclidean_distances(data_wide.T,data_wide.T)
    data_sim = data_sim / data_sim.max(axis=0).max()
    data_sim = (data_sim - data_sim.min(axis=0).min()) / (data_sim.max(axis=0).max() - data_sim.min(axis=0).min())
    data_sim = 1 - data_sim - np.eye(len(data_sim))"""

    return data_sim


def sim_bin(data_wide, lecture_list):
    data_sim = 1 - pdist(data_wide.T, 'jaccard')
    data_sim = squareform(data_sim)

    data_sim = pd.DataFrame(data_sim, index=list(data_wide), columns=list(data_wide))
    data_sim = pd.DataFrame(data_sim, index=lecture_list, columns=lecture_list)
    data_sim = data_sim.fillna(0)

    return data_sim


def sim_lecture(lecture_wide, lecture_list):
    lecture_wide["sum"] = lecture_wide.sum(axis=1)
    lecture_wide_filtered = lecture_wide[lecture_wide['sum'] != 0]
    lecture_wide_filtered = lecture_wide_filtered.drop(['sum'], axis=1)
    # 후에 행 합계가 0인 경우 제거하는 함수 발견하면 수정할 것.

    data_sim = 1 - pdist(lecture_wide_filtered, 'jaccard')
    data_sim = squareform(data_sim)

    data_sim = pd.DataFrame(data_sim, index=list(lecture_wide_filtered.T), columns=list(lecture_wide_filtered.T))
    data_sim = pd.DataFrame(data_sim, index=list(lecture_list), columns=list(lecture_list))
    data_sim = data_sim.fillna(0)
    return data_sim


def user_input(data, lecture_list):
    data_df = pd.DataFrame(data, columns=["id", "rating"])
    data_wide = data_df.pivot_table(
        columns='id',
        values='rating')
    data_wide_all = pd.DataFrame(data_wide, columns=list(lecture_list))
    data_wide_all = data_wide_all.fillna(0)

    return data_wide_all.T


#####################################################################################################

def keyword_sort(user_keyword, lecture_bin):
    keyword_index = user_keyword["keyword"]

    user_keyword = user_keyword.set_index("keyword")
    user_keyword_list = list(user_keyword.iloc[:, 0])

    check = lecture_bin * user_keyword_list
    result_list = check.sum(axis=1).sort_values(axis=0, ascending=False).index

    return result_list


def recommend(sim, input_rating):
    series = np.dot(sim, input_rating)
    result = pd.DataFrame(series, index=list(sim))
    result = result.sort_values(by=0, axis=0, ascending=False)
    print(result)
    return result.index


def classrec(request):
    try:
        data = None
        real_subjects = []
        subjects_id = []
        subjects = Subject.objects.all().order_by('subj_name')
        try:
            data = request.GET
            menu = int(data['menu'])
            keyword = str(data['keyword'])

            if menu > 0:
                menulist = [[],
                            ['일반교양', '인간과문화', '사회와역사', '자연과과학'],
                            ['확대교양', '미래융복합', '국제화', '진로와취업', '예술과체육'],
                            ['심화교양', '문학과문화', '역사와철학', '자연과생명', '기술과문명', '예술과체육']]
                menu1 = int(data['menu' + str(menu)])
                name = menulist[menu][menu1]
                if menu1 == 0:
                    subjects = Subject.objects.filter(category1__contains=name).order_by('subj_name')
                else:
                    subjects = Subject.objects.filter(category2__contains=name).order_by('subj_name')
            subjects = subjects.filter(subj_name__contains=keyword)
        except:
            pass

        user = User.objects.get(userid=request.session['userid'])
        try:
            wishlist = get_list_or_404(WishList, user_id=user.id)
            for wish in wishlist:
                subjects = subjects.exclude(subj_id=wish.subj_id)
        except:
            pass

        try:
            usersubjectlist = get_list_or_404(UserSubject, user_id=user.id)
            for usersubject in usersubjectlist:
                subjects = subjects.exclude(subj_id=usersubject.subj_id)
        except:
            pass

        for subject in subjects:
            subjects_id.append(subject.subj_id)

        recommend_subjects = []
        # 모델 연결 후 정렬 다시
        try:
            menu4 = int(data['menu4'])
            if menu4 != 0:
                # 데이터 불러오기
                user_keywords = list(UserKeyword.objects.filter(user_id=user.id).values_list('keyword_id', 'flag'))
                user_keywords = [list(x) for x in user_keywords]

                ratings = list(UserSubject.objects.values_list('user_id', 'subj_id', 'rating'))
                ratings = [list(x) for x in ratings]

                goods_single = list(UserSubject.objects.filter(user_id=user.id).values_list('subj_id', 'good'))
                goods_single = [list(x) for x in goods_single]

                goods = list(UserSubject.objects.values_list('user_id', 'subj_id', 'good'))
                goods = [list(x) for x in goods]

                wish = list(WishList.objects.values_list('user_id', 'subj_id'))
                wish = [list(x) for x in wish]

                lecture = list(SubjectKeywords.objects.values_list('subj_id', 'keyword_id', 'value'))
                lecture = [list((x[0], x[1], 1)) if x[2] > 1 else list(x) for x in lecture] #  if subjects_id.count(x[0]) > 0

                alpha = [0.7, 0.4, 0.2, 0.1]
                if menu4 == 2:
                    alpha = alpha.reverse()

                ts, lecture_list = total_sim(goods, wish, ratings, lecture, alpha)
                user = user_input(goods_single, lecture_list)
                recommend_ = recommend(ts, user)

                for i in recommend_:
                    subj = Subject.objects.get(subj_id=i)
                    recommend_subjects.append(subj)

        except:
            print('except')
            pass

        # 추천순 전환 부분
        final_subjects = subjects
        if len(recommend_subjects) > 0:
            final_subjects = recommend_subjects

        # 키워드 더하기
        for subject in final_subjects:
            try:
                if subjects_id.count(subject.subj_id) > 0:
                    keywords = get_list_or_404(SubjectKeywords.objects.exclude(value=0), subj_id=subject.subj_id)
                    keyword_list = []
                    adding_subject = subject
                    for keyword in keywords:
                        keyword_data = SubjectKeyword.objects.get(keyword_id=keyword.keyword_id)
                        keyword_list.append(keyword_data)
                    adding_subject.keywords = keyword_list
                    real_subjects.append(adding_subject)
            except:
                real_subjects.append(subject)

        paginator = Paginator(real_subjects, 5)

        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        keywords = SubjectKeyword.objects.all()

        return render(request, 'main/classrec.html', {'subjects': real_subjects, 'user': get_object_or_404(User, userid=request.session['userid'])
                                                      , 'get': data, 'page_obj': page_obj, 'keywords': keywords})
    except KeyError:
        return HttpResponseRedirect(reverse('index'))


def logout(request):
    request.session.flush()
    return HttpResponseRedirect('/')


def profile_upload(request):
    template = "main/add_subject.html"
    try:
        csv_file = request.FILES['file1']

        data_set = csv_file.read().decode('UTF-8')
        io_string = io.StringIO(data_set)
        next(io_string)
        Subject.objects.all().delete()
        for column in csv.reader(io_string, delimiter=',', quotechar='"'):
            time_room = ''
            room = ''
            try:
                time_room = column[8].split('[')
                room = time_room[1].replace(']', '')
            except:
                pass
            _, created = Subject.objects.update_or_create(
                subj_id=column[0],
                subj_name=column[2],
                category1=column[3],
                category2=column[4],
                prof_name=column[7],
                time=time_room[0],
                room=room
            )
    except:
        pass

    try:
        csv_file = request.FILES['file2']
        data_set = csv_file.read().decode('UTF-8')
        io_string = io.StringIO(data_set)
        next(io_string) 
        SubjectKeyword.objects.all().delete()
        for column in csv.reader(io_string, delimiter=','):
            _, created = SubjectKeyword.objects.update_or_create(
                keyword_id=column[1],
                keyword=column[0]
            )
    except:
        pass

    try:
        csv_file = request.FILES['file3']
        data_set = csv_file.read().decode('UTF-8')
        io_string = io.StringIO(data_set)
        next(io_string)
        SubjectKeywords.objects.all().delete()
        UserKeyword.objects.all().delete()
        for column in csv.reader(io_string, delimiter=','):
            _, created = SubjectKeywords.objects.update_or_create(
                subj_id=int(column[0]),
                keyword_id=column[1],
                value=column[2]
            )
    except:
        pass

    return render(request, template)

def myclass(request):
    try:
        user = User.objects.get(userid=request.session['userid'])
        wish_list = []
        try:
            wishlist = get_list_or_404(WishList, user_id=user.id)
            for wish in wishlist:
                subject = Subject.objects.get(subj_id=wish.subj_id)
                wish_list.append(subject)
        except:
            pass
        user_subject_list = []
        try:
            usersubjectlist = get_list_or_404(UserSubject, user_id=user.id)
            for usersubject in usersubjectlist:
                subject = Subject.objects.get(subj_id=usersubject.subj_id)
                subject.rating = usersubject.rating
                subject.good = usersubject.good
                user_subject_list.append(subject)
        except:
            pass
        real_wish_subjects = []
        for subject in wish_list:
            try:
                keywords = get_list_or_404(SubjectKeywords.objects.exclude(value=0), subj_id=subject.subj_id)
                keyword_list = []
                adding_subject = subject
                for keyword in keywords:
                    keyword_data = SubjectKeyword.objects.get(keyword_id=keyword.keyword_id)
                    keyword_list.append(keyword_data)
                adding_subject.keywords = keyword_list
                real_wish_subjects.append(adding_subject)
            except:
                real_wish_subjects.append(subject)

        real_subjects = []
        for subject in user_subject_list:
            try:
                keywords = get_list_or_404(SubjectKeywords.objects.exclude(value=0), subj_id=subject.subj_id)
                keyword_list = []
                adding_subject = subject
                for keyword in keywords:
                    keyword_data = SubjectKeyword.objects.get(keyword_id=keyword.keyword_id)
                    keyword_list.append(keyword_data)
                adding_subject.keywords = keyword_list
                real_subjects.append(adding_subject)
            except:
                real_subjects.append(subject)


        return render(request, 'main/myclass.html', {'wishlist': real_wish_subjects, 'usersubjlist': real_subjects, 'user': get_object_or_404(User, userid=request.session['userid'])})
    except KeyError:
        return HttpResponseRedirect(reverse('index'))


def wish(request, id):
    try:
        user = User.objects.get(userid=request.session['userid'])
        wish = WishList(user_id=user.id, subj_id=id)
        wish.save()
        return HttpResponseRedirect(reverse('classrec'))
    except KeyError:
        return HttpResponseRedirect(reverse('index'))


def delete_wish(request, id):
    try:
        user = User.objects.get(userid=request.session['userid'])
        wish = WishList.objects.get(user_id=user.id, subj_id=id)
        wish.delete()
        return HttpResponseRedirect(reverse('myclass'))
    except KeyError:
        return HttpResponseRedirect(reverse('index'))

def subject(request, id):
    try:
        user = User.objects.get(userid=request.session['userid'])
        good = int(request.GET['good'])
        rating = float(request.GET['grade'])
        user_subject = UserSubject(user_id=user.id, subj_id=id, good=good, rating=rating)
        user_subject.save()
        return HttpResponseRedirect(reverse('classrec'))
    except KeyError:
        return HttpResponseRedirect(reverse('index'))


def subject_(request, id):
    try:
        user = User.objects.get(userid=request.session['userid'])
        good = int(request.GET['good'])
        rating = float(request.GET['grade'])
        user_subject = UserSubject.objects.get(user_id=user.id, subj_id=id)
        user_subject.good = good
        user_subject.rating = rating
        user_subject.save()
        return HttpResponseRedirect(reverse('myclass'))
    except KeyError:
        return HttpResponseRedirect(reverse('index'))