# ohlc/urls.py

from django.contrib import admin
from django.urls import path
from ohlcapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('download-csv/', views.download_csv, name='download_csv'),
    path('visualize-chart/', views.visualize_chart, name='visualize_chart'),
]