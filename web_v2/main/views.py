import csv
import io
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import sklearn
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist, jaccard
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
from django.shortcuts import render, get_object_or_404, get_list_or_404
from django.urls import reverse
from django.core.paginator import Paginator
import json
from . import apps
from .models import *
'''import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.layers import Dense, Concatenate, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adamax'''


model = None
mu = 0

# Create your views here.
def index(request):
    return render(request, 'main/index.html')


def login(request):
    try:
        request.session['userid']
        return HttpResponseRedirect(reverse('main:class_rec_ver2'))
    except:
        return render(request, 'main/login.html')


def sign_in(request):
    global model, mu
    user_id = request.POST['user_id']
    user_pw = request.POST['user_pw']
    try:
        user = User.objects.get(userid=user_id, password=user_pw)
        request.session['userid'] = user_id
        '''if model is None: # 로그인 했을 때 모델이 비어있으면
            try:
                goods = list(UserSubject.objects.values_list('user_id', 'subj_id', 'good'))
                goods = [list(x) for x in goods]
                mbti = list(User.objects.values_list('id', 'mbti'))
                mbti = [list((x[0], int(x[1]))) for x in mbti]

                goods = pd.DataFrame(goods, columns=["id", "lecture", "rating"])
                mbti = pd.DataFrame(mbti, columns=["id", "mbti"])

                print(goods)
                print(mbti)

                mu = goods.rating.mean()  # 학습 데이터 평균
                cat_NeuralMF_begin(goods, mbti, 200)
            except Exception as ex:
                print(ex)'''
        return HttpResponseRedirect(reverse('main:class_rec_ver2'))
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
        user = User(userid=user_id, password=user_pw, username=user_name, user_number=user_number, mbti='0')
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
        return HttpResponseRedirect(reverse('main:class_rec_ver2'))

    return HttpResponseRedirect(reverse('main:signup'))


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
        return render(request, 'main/class_lec_ver2.html',
                      {'user': get_object_or_404(User, userid=request.session['userid']),
                       'user_keywords': user_keyword})
    except KeyError:
        return HttpResponseRedirect(reverse('main:index'))


def mbti(request):
    try:
        value = request.POST['hidden_mymbti']
        user = User.objects.get(userid=request.session['userid'])
        user.mbti = value
        user.save()
        return HttpResponseRedirect(reverse('main:class_rec_ver2'))
    except KeyError:
        return HttpResponseRedirect(reverse('main:class_rec_ver2'))


def keyword(request):
    try:
        user = User.objects.get(userid=request.session['userid'])
        if request.method == "POST":
            keyword_id = dict(json.loads(request.body.decode("utf-8")).items())['keyword_id']
            user_keyword = UserKeyword.objects.get(user_id=user.id, keyword_id=keyword_id)
            if user_keyword.flag == 0:
                user_keyword.flag = 1
            else:
                user_keyword.flag = 0
            user_keyword.save()
            return JsonResponse({
                'success': True, }, json_dumps_params={'ensure_ascii': True})
            # return HttpResponseRedirect(reverse('main:class_rec_ver2'))
    except KeyError:
        return JsonResponse({
            'success': False, }, json_dumps_params={'ensure_ascii': True})
        # return HttpResponseRedirect(reverse('main:class_rec_ver2'))


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
    # print(score_sim)
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
            # if(True):
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
                lecture = [list((x[0], x[1], 1)) if x[2] > 1 else list(x) for x in
                           lecture]  # if subjects_id.count(x[0]) > 0

                alpha = [0.7, 0.4, 0.4, 0.1]

                if menu4 == 2:
                    alpha.reverse()

                ts, lecture_list = total_sim(goods, wish, ratings, lecture, alpha)
                user = user_input(goods_single, lecture_list)
                recommend_ = recommend(ts, user)

                if menu4 == 3:
                    # print("야호랑이똥개")
                    # print(lecture)
                    user_keywords_wide = pd.DataFrame(user_keywords, columns=["keyword", "key"])
                    lecture_temp = pd.DataFrame(lecture, columns=["id", "keyword", "value"])
                    print(pre_lec(lecture_temp))
                    recommend_ = keyword_sort(user_keywords_wide, pre_lec(lecture_temp))
                    print(recommend_)

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

        page_number = request.GET.get('page')

        # 키워드 더하기
        i = 0
        max_i = 5
        if page_number is not None:
            i = 5 * (int(page_number) - 1)
            max_i = 5 * int(page_number)
        index = 0
        for subject in final_subjects:
            try:
                if subjects_id.count(subject.subj_id) > 0:
                    adding_subject = subject
                    if i <= index < max_i:
                        keywords = get_list_or_404(SubjectKeywords.objects.exclude(value=0), subj_id=subject.subj_id)
                        keyword_list = []
                        for keyword in keywords:
                            keyword_data = SubjectKeyword.objects.get(keyword_id=keyword.keyword_id)
                            keyword_list.append(keyword_data)
                        adding_subject.keywords = keyword_list
                    real_subjects.append(adding_subject)
                    index += 1
                else:
                    break
            except:
                real_subjects.append(subject)

        paginator = Paginator(real_subjects, 5)

        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        keywords = SubjectKeyword.objects.all()

        return render(request, 'main/class_rec_ver2.html', {
            'subjects': real_subjects,
            'user': get_object_or_404(User, userid=request.session['userid']),
            'get': data,
            'page_obj': page_obj,
            'keywords': keywords,
        })

    except KeyError:
        return HttpResponseRedirect(reverse('main:index'))


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

        return render(request, 'main/my_class_ver2.html',
                      {'wishlist': real_wish_subjects, 'usersubjlist': real_subjects,
                       'user': get_object_or_404(User, userid=request.session['userid'])})
    except KeyError:
        return HttpResponseRedirect(reverse('index'))


def wish(request, id):
    try:
        user = User.objects.get(userid=request.session['userid'])
        wish = WishList(user_id=user.id, subj_id=id)
        wish.save()
        return HttpResponseRedirect(reverse('main:class_rec_ver2'))
    except KeyError:
        return HttpResponseRedirect(reverse('main:index'))


def delete_wish(request, id):
    try:
        user = User.objects.get(userid=request.session['userid'])
        wish = WishList.objects.get(user_id=user.id, subj_id=id)
        wish.delete()
        return HttpResponseRedirect(reverse('main:my_class_ver2'))
    except KeyError:
        return HttpResponseRedirect(reverse('main:index'))


def subject(request, id):
    try:
        user = User.objects.get(userid=request.session['userid'])
        good = int(request.GET['good'])
        rating = float(request.GET['grade'])
        user_subject = UserSubject(user_id=user.id, subj_id=id, good=good, rating=rating)
        user_subject.save()
        return HttpResponseRedirect(reverse('main:class_rec_ver2'))
    except KeyError:
        return HttpResponseRedirect(reverse('main:index'))


def delete_subject(request, id):
    try:
        user = User.objects.get(userid=request.session['userid'])
        user_subject = UserSubject.objects.get(user_id=user.id, subj_id=id)
        user_subject.delete()
        return HttpResponseRedirect(reverse('main:my_class_ver2'))
    except KeyError:
        return HttpResponseRedirect(reverse('main:index'))


def subject_(request, id):
    try:
        user = User.objects.get(userid=request.session['userid'])
        good = int(request.GET['good'])
        rating = float(request.GET['grade'])
        user_subject = UserSubject.objects.get(user_id=user.id, subj_id=id)
        user_subject.good = good
        user_subject.rating = rating
        user_subject.save()
        return HttpResponseRedirect(reverse('main:my_class_ver2'))
    except KeyError:
        return HttpResponseRedirect(reverse('main:index'))


#######################################################################################################################################


def search_subject(request):
    user = User.objects.get(userid=request.session['userid'])
    user_subject_list = UserSubject.objects.filter(user_id=user.pk)
    subject_list = Subject.objects.all()
    if request.method == 'GET':
        q = request.GET.get('q', '')
        if q:
            qs = subject_list.filter(subj_name__icontains=q)
            for user_subject in user_subject_list:
                qs.exclude(subj_id=user_subject.subj_id)
        else:
            qs = None

        return render(request, 'main/search_subject.html', {
            'subject_list': qs,
        })

    return render(request, 'main/search_subject.html', {
    })


def about(request):
    return render(request, 'main/about.html', {

    })


def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def cat_NeuralMF_begin(ratings, mbti, lec_num):
    global model
    ratings = pd.merge(ratings, mbti, on='id')

    L = 16
    K = 20
    mu = ratings.rating.mean()
    M = 10001 + lec_num
    N = 10001 + lec_num

    user = Input(shape=(1,))
    item = Input(shape=(1,))
    P_embedding = Embedding(M, K, embeddings_regularizer=l2())(user)
    Q_embedding = Embedding(N, K, embeddings_regularizer=l2())(item)
    user_bias = Embedding(M, 1, embeddings_regularizer=l2())(user)
    item_bias = Embedding(N, 1, embeddings_regularizer=l2())(item)

    R = layers.dot([P_embedding, Q_embedding], axes=2)
    R = layers.add([R, user_bias, item_bias])
    R = Flatten()(R)

    P_embedding = Flatten()(P_embedding)
    Q_embedding = Flatten()(Q_embedding)
    user_bias = Flatten()(user_bias)
    item_bias = Flatten()(item_bias)
    mbti = Input(shape=(1,))
    mbti_embedding = Embedding(L, 5, embeddings_regularizer=l2())(mbti)
    mbti_layer = Flatten()(mbti_embedding)

    R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias, mbti_layer])
    R = Dense(2048)(R)
    R = Activation('linear')(R)
    R = Dense(256)(R)
    R = Activation('linear')(R)
    R = Dense(1)(R)

    model = Model(inputs=[user, item, mbti], outputs=R)
    model.compile(
        loss=RMSE,
        optimizer=SGD(),
        metrics=[RMSE]
    )
    #model.summary()

    result = model.fit(
        x=[ratings.id.values, ratings.lecture.values, ratings.mbti.values],
        y=ratings.rating.values - mu,
        epochs=80,
        batch_size=1
    )


def predict(user_id, lec_num, mbti_num):
    global model, mu
    print(model)
    user_ids = np.full((lec_num), user_id)
    mbti_ids = np.full((lec_num), mbti_num)
    item_ids = np.arange(10001, 10001 + lec_num)
    # print(item_ids)

    predictions = model.predict([user_ids, item_ids, mbti_ids]) + mu
    temp = pd.DataFrame(item_ids, columns=['lec'])
    temp['score'] = predictions
    temp = temp.sort_values(by=['score'], axis=0, ascending=False)
    print(temp)
    return list(temp['lec'])


def class_rec_ver2(request):
    user = User.objects.get(userid=request.session['userid'])

    subjects_id = []
    subjects = Subject.objects.all().order_by('subj_name')
    try:  # 찜목록 안뜨게
        wishlist = get_list_or_404(WishList, user_id=user.id)
        for wish in wishlist:
            subjects = subjects.exclude(subj_id=wish.subj_id)
    except:
        pass

    user_keyword_list = []
    try:
        keywords = SubjectKeyword.objects.all()
        try:
            user_keyword_list = get_list_or_404(UserKeyword, user_id=user.id)
        except:
            pass

        if len(keywords) > len(user_keyword_list):
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
            user_keyword_list = get_list_or_404(UserKeyword, user_id=user.id)
    except:
        pass

    keyword_name_list = []
    user_keyword = UserKeyword.objects.filter(user_id=user.id).filter(flag=1)
    for k in user_keyword:
        keyword_name_list.append(k.keyword)

    subj_keyword_list = []
    for subj in subjects:
        keywords = SubjectKeywords.objects.filter(subj_id=subj.subj_id).filter(value=1)
        li = []
        for k in keywords:
            li.append(SubjectKeyword.objects.get(keyword_id=k.keyword_id).keyword)
        subj_keyword_list.append(li)
    subj_and_keyword = list(zip(subjects, subj_keyword_list))

    if request.method == 'GET':
        print(request.GET.get('menu1', ''))
        if request.GET.get('menu1', '') == '':
            choice = 0
        else:
            choice = int(request.GET.get('menu1', ''))
        if request.GET.get('menu0', '') == '':
            menu4 = 0
        else:
            menu4 = int(request.GET.get('menu0', ''))

        menu1_list = ['전체', '일반교양', '확대교양', '심화교양']
        menu2_1 = ['전체', '인간과문화', '사회와역사', '자연과과학']
        menu2_2 = ['전체', '미래융복합', '국제화', '진로와취업', '예술과체육']
        menu2_3 = ['전체', '문학과문화', '역사와철학', '자연과생명', '기술과문명', '예술과체육']
        menu1 = menu1_list[choice]
        if request.GET.get('menu2', '') == '':
            m2 = 0
        else:
            m2 = int(request.GET.get('menu2', ''))

        if choice == 0:
            menu2 = '전체'
        elif choice == 1:
            menu2 = menu2_1[m2]
        elif choice == 2:
            menu2 = menu2_2[m2]
        elif choice == 3:
            menu2 = menu2_3[m2]

        for subject in subjects:
            subjects_id.append(subject.subj_id)

        recommend_subjects = []

        try:
            if menu4 != 0:
                print('만족도 기반')
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
                mbti = list(User.objects.values_list('id', 'mbti'))
                mbti = [list(x) for x in mbti]
                lecture = list(SubjectKeywords.objects.values_list('subj_id', 'keyword_id', 'value'))
                lecture = [list((x[0], x[1], 1)) if x[2] > 1 else list(x) for x in
                           lecture]  # if subjects_id.count(x[0]) > 0

                # MF 모델
                '''predict(15, 200, 0)

                recommend_ = []'''

                # 기존 모델
                alpha = [0.7, 0.4, 0.4, 0.1]

                if menu4 == 2:
                    alpha.reverse()
                ts, lecture_list = total_sim(goods, wish, ratings, lecture, alpha)
                if menu4 == 3:
                    # print(lecture)
                    user_keywords_wide = pd.DataFrame(user_keywords, columns=["keyword", "key"])
                    lecture_temp = pd.DataFrame(lecture, columns=["id", "keyword", "value"])
                    print(pre_lec(lecture_temp))
                    recommend_ = keyword_sort(user_keywords_wide, pre_lec(lecture_temp))
                    print(recommend_)
                else:
                    user = user_input(goods_single, lecture_list)
                    recommend_ = recommend(ts, user)

                subj = Subject.objects.all()
                if menu1 != '전체':
                    subj = Subject.objects.filter(category1__contains=menu1).order_by('subj_name')
                    if menu2 != '전체':
                        subj = Subject.objects.filter(category2__contains=menu2).order_by('subj_name')
                # 들었던 강의 제외
                try:
                    usersubjectlist = UserSubject.objects.filter(
                        user_id=User.objects.get(userid=request.session['userid']).id)
                    for usersubject in usersubjectlist:
                        subj = subj.exclude(subj_id=usersubject.subj_id)
                except:
                    pass

                subj_id_list = []
                subj_keyword_list = []
                for x in subj:
                    subj_id_list.append(x.subj_id)
                for i in recommend_:
                    if i in subj_id_list:
                        subj = Subject.objects.get(subj_id=i)
                        subj_keyword = SubjectKeywords.objects.filter(subj_id=i).filter(value=1)
                        li = []
                        for k in subj_keyword:
                            li.append(SubjectKeyword.objects.get(keyword_id=k.keyword_id).keyword)
                        subj_keyword_list.append(li)
                        recommend_subjects.append(subj)
                subj_and_keyword = list(zip(recommend_subjects, subj_keyword_list))
        except Exception as ex:
            print(ex.__str__())
            pass

        paginator = Paginator(subj_and_keyword, 5)

        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        return render(request, 'main/class_rec_ver2.html', {
            'user': User.objects.get(userid=request.session['userid']),
            'keyword_name_list': keyword_name_list,
            'subj_and_keyword': subj_and_keyword,
            'user_keyword_list': user_keyword_list,
            'page_obj': page_obj,
        })

    if request.method == 'POST':
        user_keyword = UserKeyword.objects.filter(user_id=user.id)
        keyowrd = dict(json.loads(request.body.decode("utf-8")).items())
        if user_keyword:
            for keyword in user_keyword:
                if keyword.keyword in keyowrd['keyword_list']:
                    keyword.flag = 1
                    keyword.save()
                else:
                    keyword.flag = 0
                    keyword.save()
        else:
            subject_keyword = SubjectKeyword.objects.all()
            for keyword in subject_keyword:
                if keyword.keyword in keyowrd['keyword_list']:
                    flag = 1
                else:
                    flag = 0
                user_keyword = UserKeyword.objects.create(
                    user_id=user.id,
                    keyword_id=keyword.keyword_id,
                    keyword=keyword.keyword,
                    flag=flag
                )
                user_keyword.save()

    return render(request, 'main/class_rec_ver2.html', {
        'user': user,
        'keyword_name_list': keyword_name_list,
        'subj_and_keyword': subj_and_keyword,
        'user_keyword_list': user_keyword_list,
    })


def my_class_ver2(request):
    user = User.objects.get(userid=request.session['userid'])
    user_subject_list = UserSubject.objects.filter(user_id=user.id)
    wish_list = WishList.objects.filter(user_id=user.id)
    pre_lec_list = []
    save_lec_list = []
    for user_subject in user_subject_list:
        pre_lec_list.append(Subject.objects.get(subj_id=user_subject.subj_id))
    for lec in wish_list:
        save_lec_list.append(Subject.objects.get(subj_id=lec.subj_id))
    return render(request, 'main/my_class_ver2.html', {
        'user': user,
        'pre_lec_list': pre_lec_list,
        'save_lec_list': save_lec_list,
    })


def pre_lec_ver2(request):
    user = User.objects.get(userid=request.session['userid'])
    user_subject_list = UserSubject.objects.filter(user_id=user.id)
    subject_list = Subject.objects.all()
    keyword_list = SubjectKeyword.objects.all()
    q = ''
    if request.method == 'GET':
        q = request.GET.get('q', '')
        if q:
            qs = subject_list.filter(subj_name__icontains=q)
            for user_subject in user_subject_list:
                qs = qs.exclude(subj_id=user_subject.subj_id)
        else:
            qs = None
    elif request.method == 'POST':
        pre_lec = dict(json.loads(request.body.decode("utf-8")).items())
        print(pre_lec)
        user_id = User.objects.get(userid=request.session['userid']).id
        subj_id = pre_lec['subject']
        good = int(pre_lec['good'])
        rating = float(pre_lec['rating'])
        keyword_list = pre_lec['keyword_list']
        for keyword in keyword_list:
            keyword = SubjectKeyword.objects.get(keyword=keyword)
            keywords = SubjectKeywords.objects.get(subj_id=subj_id, keyword_id=keyword.keyword_id)
            keywords.value = keywords.value + 1
            keywords.save()

        user_subject_new = UserSubject(user_id=user_id, subj_id=subj_id, good=good, rating=rating)
        print(user_subject_new.__dict__)
        user_subject_new.save()
        q = pre_lec['search']
        if q:
            qs = subject_list.filter(subj_name__icontains=q)
            for user_subject in user_subject_list:
                qs = qs.exclude(subj_id=user_subject.subj_id)
        else:
            qs = None

    return render(request, 'main/pre_lec_ver2.html', {
        'user': user,
        'search': q,
        'subject_list': qs,
        'keyword_list': keyword_list,
    })


def pre_lec_delete(request):
    if request.method == 'POST':
        pre_lec = dict(json.loads(request.body.decode("utf-8")).items())
        subj_id = pre_lec['subj_id']
        user = User.objects.get(userid=request.session['userid'])
        user_subject = UserSubject.objects.get(user_id=user.id, subj_id=subj_id)
        user_subject.delete()
        return HttpResponseRedirect(reverse('main:my_class_ver2'))


def save_lec_delete(request):
    if request.method == 'POST':
        save_lec = dict(json.loads(request.body.decode("utf-8")).items())
        subj_id = save_lec['subj_id']
        user = User.objects.get(userid=request.session['userid'])
        try:
            wish_list = get_list_or_404(WishList.objects.all(), user_id=user.id, subj_id=subj_id)
            for wish in wish_list:
                wish.delete()
        except:
            pass

        return HttpResponseRedirect(reverse('main:my_class_ver2'))
