FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html styles.css app.js favicon.svg /usr/share/nginx/html/

EXPOSE 80
