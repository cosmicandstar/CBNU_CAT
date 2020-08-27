from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login, name='login'),
    path('signin/', views.sign_in, name='signin'),
    path('signup/', views.sign_up, name='signup'),
    path('join/', views.join, name='join'),
    path('myinfo/', views.myinfo, name='myinfo'),
    path('myinfo/mbti', views.mbti, name='mbti'),
    path('myinfo/keyword/<int:id>/', views.keyword, name='keyword'),
    path('myinfo/-keyword/<int:id>/', views.keyword, name='-keyword'),
    path('logout/', views.logout, name='logout'),
    path('classrec/', views.classrec, name='classrec'),
    path('upload-csv/', views.profile_upload, name="profile_upload"),
    path('myclass/', views.myclass, name='myclass'),
    path('classrec/wish/<int:id>/', views.wish, name='wish'),
    path('myclass/-wish/<int:id>/', views.delete_wish, name='-wish'),
    path('classrec/subject/<int:id>/', views.subject, name='subject'),
    path('myclass/subject_/<int:id>/', views.subject_, name='subject_'),
    path('myclass/-subject/<int:id>/', views.delete_subject, name='-subject'),
]
