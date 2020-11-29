from django.contrib import admin

from main.models import *


# Register your models here.
admin.site.register(User)
admin.site.register(Subject)
admin.site.register(UserKeyword)
admin.site.register(WishList)
admin.site.register(UserSubject)
admin.site.register(SubjectKeywords)
admin.site.register(SubjectKeyword)