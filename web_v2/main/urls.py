from django.urls import path
from . import views


app_name = 'main'


urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login, name='login'),
    path('signin/', views.sign_in, name='signin'),
    path('signup/', views.sign_up, name='signup'),
    path('join/', views.join, name='join'),

    path('myinfo/', views.myinfo, name='myinfo'),
    path('class_rec_ver2/mbti', views.mbti, name='mbti'),
    path('class_rec_ver2/keyword/', views.keyword, name='keyword'),
    path('class_rec_ver2/-keyword/', views.keyword, name='-keyword'),

    path('logout/', views.logout, name='logout'),
    path('classrec/', views.classrec, name='classrec'),
    path('upload-csv/', views.profile_upload, name="profile_upload"),
    path('myclass/', views.myclass, name='myclass'),
    path('class_rec_ver2/wish/<int:id>/', views.wish, name='wish'),
    path('myclass/-wish/<int:id>/', views.delete_wish, name='-wish'),
    path('classrec/subject/<int:id>/', views.subject, name='subject'),
    path('myclass/subject_/<int:id>/', views.subject_, name='subject_'),
    path('myclass/-subject/<int:id>/', views.delete_subject, name='-subject'),
    
    path('search_subject', views.search_subject, name='search_subject'),
    path('about', views.about, name='about'),
    path('class_rec_ver2', views.class_rec_ver2, name='class_rec_ver2'),
    path('my_class_ver2', views.my_class_ver2, name='my_class_ver2'),
    path('pre_lec_ver2', views.pre_lec_ver2, name='pre_lec_ver2'),
    path('my_class_ver2/pre_lec/delete', views.pre_lec_delete, name='pre_lec_delete'),
    path('my_class_ver2/save_lec/delete', views.save_lec_delete, name='save_lec_delete'),
]
