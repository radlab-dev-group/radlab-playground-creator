"""
URL configuration for transped project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from django.conf import settings
from rest_framework.authtoken.views import obtain_auth_token

urlpatterns = []

if settings.SYSTEM_HANDLER.show_admin_window():
    from django.contrib import admin

    urlpatterns.append(path("admin/", admin.site.urls))

if settings.SYSTEM_HANDLER.use_keycloak_authentication():
    from keycloak import urls as keycloak_urls

    urlpatterns += keycloak_urls.urlpatterns
else:
    urlpatterns.append(
        path("api-token-auth/", obtain_auth_token, name="api_token_auth")
    )

# Auto added urls during installation
# article_creator_urls.py
from creator.urls import urlpatterns as article_creator_urlpatterns

urlpatterns += article_creator_urlpatterns

# system_urls.py
from system.urls import urlpatterns as system_creator_urlpatterns

urlpatterns += system_creator_urlpatterns


# End of auto added urls
