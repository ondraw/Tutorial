from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
import songs.routing

application = ProtocolTypeRouter({
    'websocket': AuthMiddlewareStack(URLRouter(songs.routing.websocket_urlpatterns))
})