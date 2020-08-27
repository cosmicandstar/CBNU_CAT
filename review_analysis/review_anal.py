import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from operator import itemgetter


def get_NVM(text):
    tokenizer = MeCab.Tagger()
    parsed = tokenizer.parse(text)
    word_tag = [w for w in parsed.split("\n")]
    pos = []
    tags = ['NNG', 'NNP', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'NR', 'SL']
    for word_ in word_tag[:-2]:
        word = word_.split("\t")
        tag = word[1].split(",")[0]
        if tag in tags:
            pos.append(word[0])
        elif '+' in tag:
            if 'VV' in tag or 'VA' in tag or 'VX' in tag:
                t = word[1].split(',')[-1].split('/')[0]
                pos.append(t)
    return pos

kwards = {'객관식': ['객관식', '객관식 있', '객관식 시험', '주관식 아니'],
          '주관식': ['주관식', '주관식 있', '약술', '주관식 시험', '객관식 아니', '단답', '서술형', '논술형'],
          '프로젝트 있음': ['프로젝트', '텀 과제', 'project'],
          '암기 많음': ['암기', '외우', '암기 있']}

if __name__ == '__main__':

    # f = pd.read_csv("./reviews_row.csv", encoding='utf-8')
    # cnt_vect_by_1 = CountVectorizer(tokenizer=get_NVM, ngram_range=(1, 1), min_df=1)
    # temp = f.drop_duplicates(['순번'])
    # idx = temp['순번']
    # col = []
    # rows = []
    # for i in idx:
    #     lec_feature = {
    #         '순번': 0,
    #         'lecture': '',
    #         'professor': '',
    #         '과제 많음': 0, '과제 보통': 0, '과제 없음': 0,
    #         '조모임 많음': 0, '조모임 보통': 0, '조모임 없음': 0,
    #         '학점 느님': 0, '학점 비율채워줌': 0, '학점 깐깐함': 0, 'F폭격기': 0,
    #         '출결 혼용': 0, '출결 직접호명': 0, '출결 지정좌석': 0, '전자출결': 0, '출결 반영안함': 0,
    #         '시험 네 번 이상': 0, '시험 세 번': 0, '시험 두 번': 0, '시험 한 번': 0, '시험 없음': 0,
    #         'pass/fail': 0, 'grade': 0,
    #         '객관식': 0,
    #         '주관식': 0,
    #         '프로젝트 있음': 0,
    #         '암기 많음': 0,
    #         '리뷰 수': 0}
    #     t = f['순번'] == i
    #     temp = f[t]
    #     li = list(zip(list(temp['순번']), list(temp['과목명']), list(temp['담당교수']), list(temp['성적부여']), list(temp['평점']), list(temp['과제']), list(temp['조모임']), list(temp['학점 비율']), list(temp['출결']), list(temp['시험 횟수']), list(temp['추천수']), list(temp['리뷰'])))
    #     review_cnt = len(li)
    #     lec_feature['순번'] = int(li[0][0])
    #     lec_feature['lecture'] = li[0][1]
    #     lec_feature['professor'] = li[0][2]
    #     try:
    #         if '많음' in li[0][5]:
    #             lec_feature['과제 많음'] = 1
    #         elif '보통' in li[0][5]:
    #             lec_feature['과제 보통'] = 1
    #         elif '없음' in li[0][5]:
    #             lec_feature['과제 없음'] = 1
    #
    #         if '많음' in li[0][6]:
    #             lec_feature['조모임 많음'] = 1
    #         elif '보통' in li[0][6]:
    #             lec_feature['조모임 보통'] = 1
    #         elif '없음' in li[0][6]:
    #             lec_feature['조모임 없음'] = 1
    #
    #         if '느님' in li[0][7]:
    #             lec_feature['학점 느님'] = 1
    #         elif '비율' in li[0][7]:
    #             lec_feature['학점 비율채워줌'] = 1
    #         elif '깐깐' in li[0][7]:
    #             lec_feature['학점 깐깐함'] = 1
    #         elif 'F' in li[0][7]:
    #             lec_feature['F폭격기'] = 1
    #
    #         if '혼용' in li[0][8]:
    #             lec_feature['출결 혼용'] = 1
    #         elif '직접호명' in li[0][8]:
    #             lec_feature['출결 직접호명'] = 1
    #         elif '지정좌석' in li[0][8]:
    #             lec_feature['출결 지정좌석'] = 1
    #         elif '전자' in li[0][8]:
    #             lec_feature['전자출결'] = 1
    #         elif '반영안함' in li[0][8]:
    #             lec_feature['출결 반영안함'] = 1
    #
    #         if '없음' in li[0][9]:
    #             lec_feature['시험 없음'] = 1
    #         elif '한 번' in li[0][9]:
    #             lec_feature['시험 한 번'] = 1
    #         elif '두 번' in li[0][9]:
    #             lec_feature['시험 두 번'] = 1
    #         elif '세 번' in li[0][9]:
    #             lec_feature['시험 세 번'] = 1
    #         elif '네 번' in li[0][9]:
    #             lec_feature['시험 네 번 이상'] = 1
    #
    #         if 'pass' in li[0][3]:
    #             lec_feature['pass/fail'] = 1
    #         elif 'Grade' in li[0][3]:
    #             lec_feature['grade'] = 1
    #     except:
    #         pass
    #     for x in li:
    #         try:
    #             re1 = x[11] + ' ' + (int(x[10]) * x[11])
    #             review_cnt += int(x[10])
    #         except:
    #             re1 = ''
    #
    #         df = pd.DataFrame([re1], columns=['review'])
    #         try:
    #             dtm = cnt_vect_by_1.fit_transform(df['review'])
    #         except:
    #             continue
    #         voca = {}
    #         for idx, word in enumerate(cnt_vect_by_1.get_feature_names()):
    #             voca[word] = dtm.getcol(idx).sum()
    #         words = sorted(voca.items(), key=lambda x: x[1], reverse=True)
    #         for word_ in words:
    #             for key in kwards.keys():
    #                 for value in kwards[key]:
    #                     if value in word_[0]:
    #                         lec_feature[key] += word_[1]
    #         # 순번0,과목명1,담당교수2,성적부여3,평점4,과제5,조모임6,학점 비율7,출결8,시험 횟수9,추천수10,리뷰11
    #
    #     al = 0.05
    #     if review_cnt * al < int(lec_feature['객관식']): lec_feature['객관식'] = 1
    #     else: lec_feature['객관식'] = 0
    #     if review_cnt * al < int(lec_feature['주관식']): lec_feature['주관식'] = 1
    #     else: lec_feature['주관식'] = 0
    #     if review_cnt * al < int(lec_feature['프로젝트 있음']): lec_feature['프로젝트 있음'] = 1
    #     else: lec_feature['프로젝트 있음'] = 0
    #     if review_cnt * al < int(lec_feature['암기 많음']): lec_feature['암기 많음'] = 1
    #     else: lec_feature['암기 많음'] = 0
    #     lec_feature['리뷰 수'] = review_cnt
    #
    #     col = list(lec_feature.keys())
    #     row = list(lec_feature.values())
    #     rows.append(row)
    # rows.insert(0, col)
    # df = pd.DataFrame(rows)
    # df.to_csv("./lec_anal_0.05.csv", index=False, header=False, encoding='utf-8-sig')
