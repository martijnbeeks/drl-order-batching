class Event:
    ARRIVAL = 0
    DEPARTURE = 1

    def __init__(self, typ, station, time, order):  # type is a reserved word
        self.type = typ
        self.station = station
        self.time = time
        self.order = order

    def __str__(self):
        s = ('Arrival', 'Departure')
        return s[self.type] + " at station " + str(self.station) + ' at t = ' + str(self.time) + ' of order ' + str(
            self.order.ID)

    def __lt__(self, other):
        return self.time < other.time


