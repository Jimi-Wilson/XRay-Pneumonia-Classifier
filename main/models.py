from django.db import models

classes = ((0, ('Normal')), (1, ('Pneumonia')))


class Prediction(models.Model):
    image = models.ImageField(upload_to='x-rays')
    prediction = models.CharField(max_length=200, choices=classes, null=True, blank=True)
