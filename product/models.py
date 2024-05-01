from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class LogReport(models.Model):
   
    user = models.ForeignKey(User, related_name="user_log", on_delete=models.CASCADE, null = True, blank=True)
    dateTime = models.DateTimeField(auto_now=True)
    product = models.CharField(max_length=200)
 

    def ___str__(self):
        return str(self.user.id)
