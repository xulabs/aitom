from django.db import models
from django2_resumable.fields import ResumableFileField


class Document(models.Model):
    # input form for resumeable.js
    # description = models.CharField(max_length=255, blank=True)
    # document = models.FileField(upload_to='')
    document = ResumableFileField(chunks_upload_to='mrc/')

    # can optionally add the timestamp of file upload
    # uploaded_at = models.DateTimeField(auto_now_add=True)


class zoomInput(models.Model):
    # define a form with 6 input fields for coordinates
    luX = models.IntegerField(default=0)
    luY = models.IntegerField(default=0)
    luZ = models.IntegerField(default=0)

    rdX = models.IntegerField(default=0)
    rdY = models.IntegerField(default=0)
    rdZ = models.IntegerField(default=0)


class sliceInput(models.Model):
    # define a form with 1 select for default plane 6 input fields for coordinates
    Default_Plane = [
        ('xoy', 'XOY'),
        ('yoz', 'YOZ'),
        ('xoz', 'XOZ')
    ]
    plane = models.CharField(max_length=3, choices=Default_Plane, default='XOY')

    cX = models.IntegerField(default=0)
    cY = models.IntegerField(default=0)
    cZ = models.IntegerField(default=0)

    rX = models.IntegerField(default=0)
    rY = models.IntegerField(default=0)
    rZ = models.IntegerField(default=0)
