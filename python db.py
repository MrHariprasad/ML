'''
import mysql.connector

mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    password="123456")
print(mydb)

import mysql.connector

cnx = mysql.connector.connect(user='root', password='123456',
                              host='127.0.0.1',database='paramesh',
                              charset='utf8')
cnx.close()
import mysql.connector
'''
import mysql.connector
# Establish connection
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="123456",
    database="myDB",
    charset="utf8"
)

# Create a cursor object
mycursor = mydb.cursor()

# Execute SQL query
mycursor.execute("SELECT * FROM MyGuests")

# Fetch all results
myresult = mycursor.fetchall()

# Print each row
for x in myresult:
    print(x)
