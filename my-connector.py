import mysql.connector

mydb=mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    port='3306',
    database='test'
)

mycursor=mydb.cursor()
mycursor.execute('select * from numberplate')
list = mycursor.fetchall()
print(list)

mycursor.execute('insert into test.numberplate(Num) values (#{extract_text.text)  ')

