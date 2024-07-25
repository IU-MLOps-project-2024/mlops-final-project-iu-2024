docker build -t category_model -f api/Dockerfile api/
docker run -d -p 5152:8080 --name category_container category_model

docker login
docker tag category_model datapaf/category_model:latest
docker push datapaf/category_model:latest
