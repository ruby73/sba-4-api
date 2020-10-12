from com_sba_api.food.food_api import FoodsApi
def initialize_routes(api):
    api.add_resource(FoodsApi, '/api/foods')