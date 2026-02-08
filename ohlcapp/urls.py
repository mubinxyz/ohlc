# ohlcapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('download-csv/', views.download_csv, name='download_csv'),
    path('visualize-chart/', views.visualize_chart, name='visualize_chart'),
]