FROM python:3.12

#set the working dir
WORKDIR /usr/src/app

# Copy all the files to the container
COPY . .

#install dependencies
RUN pip install -r requirements.txt

#exposing the port
EXPOSE 5000

#run the app
CMD ["python", "app.py"]