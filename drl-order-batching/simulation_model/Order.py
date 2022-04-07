class Order:
    ID = 0

    def __init__(self, arr_time, cutoff_time, nOrders, nItems_ptg, nItems_gtp, route, order_category, action):
        self.ID = Order.ID
        self.arr_time = arr_time
        self.cutoff_time = cutoff_time
        self.nOrders = nOrders
        self.nItems_ptg = nItems_ptg
        self.nItems_gtp = nItems_gtp
        self.route = route
        self.category = order_category
        self.action = action
        Order.ID += 1

        self.PtG_in = 0
        self.PtG_out = 0

        self.GtP_in = 0
        self.GtP_out = 0

        self.Pack_in = 0
        self.Pack_out = 0

        self.DtO_in = 0
        self.DtO_out = 0

        self.StO_in = 0
        self.StO_out = 0

        self.System_out = 0

    def __str__(self):
        return self.ID


