import os
import pymongo
import sys

# Replace the placeholder data with your Atlas connection string. Be sure it includes
# a valid username and password! Note that in a production environment,
# you should not store your password in plain-text here.
def get_database():
    try:
        client = pymongo.MongoClient(os.getenv('MONGODB_URL'))
        print('success')
    # return a friendly error if a URI error is thrown 
    except pymongo.errors.ConfigurationError:
        print("An Invalid URI host error was received. Is your Atlas host name correct in your connection string?")
        sys.exit(1)

    # use a database named "myDatabase"
    db = client.myDatabase
    return db

