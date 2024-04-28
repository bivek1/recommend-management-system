from django.urls import path
from .import views


app_name = "product"


urlpatterns =[
    path('', views.homepage, name ="homepage"),
    path('details/<int:id>', views.productView, name ="detail"),
    path('filter-bike', views.filterBike, name="bike"),
    path('filter-clothes', views.filterClothes, name="cloth"),
    path('filter-book', views.filterBook, name="book"),
]