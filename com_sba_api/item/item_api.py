from flask_restful import Resource
from flask import Response, jsonify
from com_sba_api.item.item_dao import ItemDao

class ItemsApi(Resource):

    def __init__(self):
        self.dao = ItemDao()

    def get(self):
       items = self.dao.select_items()
       return jsonify(items[0])
        