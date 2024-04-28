from django.db import models

# Create your models here.
class LogReport(models.Model):
    name = models.CharField(max_length=200)
    device = models.CharField(max_length=200)
    dateTime = models.DateTimeField(auto_now=True)
    product = models.CharField(max_length=200)
    session = models.CharField(max_length=100)

    def ___str__(self):
        return self.name
