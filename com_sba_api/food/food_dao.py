import mysql.connector
from com_sba_api.ext.db import config


class FoodDao:
    
    def __init__(self):
        self.connector  = mysql.connector.connect(**config)
        self.cursor = self.connector.cursor(dictionary=True)


    def select_foods(self):
        cur = self.cursor
        con = self.connector
        rows = []
        try:
            cur.execute('select * from food',)
            rows = cur.fetchall()
            for row in rows:
                print(f'price is : {str(row["price"])}')
            
            cur.close()
        except:
            print('Exception ...')

        finally:
            if con is not None:
                con.close()
        return rows

print('---2---')
dao = FoodDao()
dao.select_foods()