from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('AdminLogin.html', views.AdminLogin, name="AdminLogin"), 
	       path('AdminLoginAction', views.AdminLoginAction, name="AdminLoginAction"),
	       path('LoadDataset', views.LoadDataset, name="LoadDataset"),
	       path('RunLOF', views.RunLOF, name="RunLOF"),
	       path('LoadDatasetAction', views.LoadDatasetAction, name="LoadDatasetAction"),	   
	       path('RunIsolation', views.RunIsolation, name="RunIsolation"),
	       path('RunOCS', views.RunOCS, name="RunOCS"),
	       path('RunPaacda', views.RunPaacda, name="RunPaacda"),	
	       path('RunExtension', views.RunExtension, name="RunExtension"), 	   
	       path('Aboutus', views.Aboutus, name="Aboutus"), 	   
]