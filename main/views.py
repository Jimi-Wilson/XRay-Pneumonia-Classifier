from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from django.views import View
from django.views.generic.edit import CreateView
from .models import Prediction
from predict import predict


class PredictView(CreateView):
    model = Prediction
    fields = ['image']
    template_name = 'main/predict.html'

    def form_valid(self, form):
        self.object = form.save()
        img_path = self.object.image.path
        pred = predict(img_path)
        self.object.prediction = pred
        self.object.save()
        return HttpResponseRedirect(self.get_success_url())

    def get_success_url(self):
        return reverse('prediction', kwargs={'pk': self.object.pk})


class PredictionView(View):
    template_name = 'main/prediction.html'
    fail_template_name = 'main/fail.html'

    def get(self, request, *args, **kwargs):

        try:
            pk = kwargs.get('pk')
            prediction = Prediction.objects.all().get(id=pk)
            class_dict = {0: 'Normal', 1: 'Pneumonia'}
            context = {'data': prediction, 'class_name': class_dict[int(prediction.prediction)]}
            return render(request, self.template_name, context)
        except ObjectDoesNotExist:
            return render(request, self.fail_template_name)



