from django import forms
from .models import Document, zoomInput, sliceInput


# Model Form for .mrc file upload, implemented using resumable js for django extension
class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('document',)

    # Model Form for inputting 6 coordinates for zoom, in the order left up x,y,z and bottom down x,y,z


class zoomForm(forms.ModelForm):
    class Meta:
        model = zoomInput
        fields = ('luX', 'luY', 'luZ', 'rdX', 'rdY', 'rdZ',)


class sliceForm(forms.ModelForm):
    class Meta:
        model = sliceInput
        fields = ('plane', 'cX', 'cY', 'cZ', 'rX', 'rY', 'rZ',)
