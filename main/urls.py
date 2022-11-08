from django.urls import path
from .views import *

urlpatterns = [
    path('predict', PredictView.as_view() , name='predict'),
    path('prediction/<int:pk>', PredictionView.as_view() , name='prediction')
]