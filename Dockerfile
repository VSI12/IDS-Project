FROM python:3.12

#set the working dir
WORKDIR /usr/src/app

#copy the requirements
COPY requirements.txt .
#install dependencies
RUN pip install -r requirements.txt

# Copy all the files to the container
COPY . .


#exposing the port
EXPOSE 5000

#run the app
CMD ["python", "app.py"]