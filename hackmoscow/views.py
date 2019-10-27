# coding=utf-8
from django.shortcuts import render
from django.views.generic import TemplateView

from recommendation import parse_tracks_from_source, get_recommendation


class HomeView(TemplateView):
    template_name = 'index.html'

    def post(self, request, *args, **kwargs):
        link = request.POST.get('link')
        source = 'youtube' if 'youtube' in link else 'yandex'
        tracks = parse_tracks_from_source(source, link)
        titles, images, distances = get_recommendation(tracks)
        recommends = []
        for i in xrange(len(titles)):
            recommends.append(dict(title=titles[i], image=images[i], distance=round(distances[i], 2)))
        return render(request, 'index.html', dict(recommends=recommends, source=source, link=link))


class JenreView(TemplateView):
    template_name = 'jenres.html'

class AudioView(TemplateView):
    template_name = 'audio.html'
