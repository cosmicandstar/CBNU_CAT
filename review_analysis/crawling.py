from bs4 import BeautifulSoup
import requests
import re
from selenium import webdriver
import pandas as pd
from operator import itemgetter
import os
import pandas as pd


ID = ''
PWD = ''


# 강의명 검색
def search(lec_name):
    driver.get('https://chungbuk.everytime.kr/lecture')
    search = driver.find_element_by_name('keyword')
    search.clear()
    search.send_keys(lec_name)
    search.submit()

# 강의 정보 및 리뷰 추출
def get_review_by_lec():
    lectures = []
    for i in range(len(driver.find_elements_by_xpath('//div[@class="lectures"]/a[@class="lecture"]'))):
        lecture_info = {}
        driver.find_elements_by_xpath('//div[@class="lectures"]/a[@class="lecture"]')[i].click()
        # 과목명_교수
        try:
            head = driver.find_element_by_xpath('//div[@class="side head"]')
            lecture_info['lecture'] = head.find_element_by_tag_name('h2').text
            lecture_info['professor'] = head.find_element_by_tag_name('span').text
            # 강의 정보
            rating = driver.find_element_by_xpath('//div[@class="rating"]')
            lecture_info['star'] = rating.find_element_by_xpath('//span[@class="value"]').text
            for p in rating.find_elements_by_xpath('//div[@class="details"]/p'):
                lecture_info[p.find_element_by_tag_name('label').text] = p.find_element_by_tag_name('span').text

            # 리뷰
            reviews = []
            for article in driver.find_elements_by_xpath('//div[@class="articles"]/article'):
                review_detail = {}
                try:
                    review_detail['review_posvote'] = article.find_elements_by_xpath('./p/span')[2].text
                except:
                    review_detail['review_posvote'] = '0'
                review_detail['review'] = article.find_element_by_class_name("text").text.replace('\n', ' ')
                reviews.append(review_detail)
            lectures.append((lecture_info, reviews))
            driver.back()
            driver.implicitly_wait(1)
        except:
            pass

    return lectures

def crawling():
    driver = webdriver.Chrome('C:/Users/seongjin/Desktop/chromedriver_win32/chromedriver')
    driver.implicitly_wait(3)
    driver.get('https://chungbuk.everytime.kr/login')

    # 로그인
    driver.find_element_by_name('userid').send_keys(ID)
    driver.find_element_by_name('password').send_keys(PWD)
    driver.find_element_by_xpath('//p[@class="submit"]/input').click()

    lecture_csv = pd.read_csv('./Lecture_list(2020).csv', encoding='utf-8')
    lecture_set = set(lecture_csv['과목명'])

    lec_row = []
    review_row = []
    for lecture_name in lecture_set:
        try:
            search(lecture_name)
            lectures = get_review_by_lec()
        except:
            pass

        for i in range(len(lectures)):
            lecture_info, reviews = lectures[i]
            lec_row.append(list(lecture_info.values()))

            lecture, professor = lecture_info['lecture'], lecture_info['professor']
            temp = []
            temp.append(lecture)
            temp.append(professor)
            for review in reviews:
                review_row.append(temp + list(review.values()))

    lec_df = pd.DataFrame(lec_row, columns=['lecture', 'professor', 'star', '과제', '조모임', '학점 비율', '출결', '시험 횟수'])
    lec_df.to_csv("./lecture_info.csv", encoding='utf-8-sig')

    review_df = pd.DataFrame(review_row, columns=['lecture', 'professor', 'review_posvote', 'review'])
    review_df.to_csv("./reviews.csv", encoding='utf-8-sig')
