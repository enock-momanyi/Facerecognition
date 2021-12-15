from django.shortcuts import render
from django.http import HttpResponse

from app.forms import FaceRecognitionForm
from app.machinelearning import pipeline_model
from django.conf import settings
from app.forms import FaceRecognition
import os
# Create your views here.

def index(request):
    form = FaceRecognitionForm(request.POST or None, request.FILES or None)
    if request.method == 'POST':
        if form.is_valid():
            save = form.save(commit=True)
            primary_key = save.pk
            imgobj = FaceRecognition.objects.get(pk=primary_key)
            fileroot = str(imgobj.image)
            filepath = os.path.join(settings.MEDIA_ROOT, fileroot)
            results = pipeline_model(filepath)
        return render(request, 'index.html',{'form':form, 'upload':True, 'results':results})
            
    return render(request, 'index.html',{'form':form, 'upload':False})
