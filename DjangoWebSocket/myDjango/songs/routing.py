from django.conf.urls import url
from . import testws

websocket_urlpatterns = [
    url(r'^ws/songs/(?P<room_name>[^/]+)/$', testws.TestWS.as_asgi()),
]