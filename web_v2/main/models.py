from django.contrib.auth.base_user import AbstractBaseUser
from django.contrib.auth.models import UserManager
from django.db import models


# Create your models here.
class User(AbstractBaseUser):
    userid = models.TextField(max_length=20) # 계정
    username = models.TextField(max_length=20)
    user_number = models.TextField(max_length=10)
    mbti = models.TextField(max_length=4)

    objects = UserManager()

    USERNAME_FIELD = 'userid'
    REQUIRED_FIELDS = ['user_number']

    def __str__(self):
        return self.username


class Subject(models.Model):
    subj_id = models.IntegerField(unique=True)
    subj_name = models.TextField(max_length=20)
    category1 = models.TextField(max_length=10)
    category2 = models.TextField(max_length=10)
    prof_name = models.TextField(max_length=10)
    grading = models.TextField(max_length=100)
    time = models.TextField(max_length=20)
    room = models.TextField(max_length=10)


class SubjectKeywords(models.Model):
    subj_id = models.IntegerField()
    keyword_id = models.IntegerField()
    value = models.IntegerField()


class SubjectKeyword(models.Model):
    keyword_id = models.IntegerField()
    keyword = models.TextField(max_length=10)


class UserKeyword(models.Model):
    user_id = models.IntegerField()
    keyword_id = models.IntegerField()
    keyword = models.TextField(max_length=10)
    flag = models.IntegerField()

class WishList(models.Model):
    user_id = models.IntegerField()
    subj_id = models.IntegerField()


class UserSubject(models.Model):
    user_id = models.IntegerField()
    subj_id = models.IntegerField()
    good = models.IntegerField()
    rating = models.FloatField()


