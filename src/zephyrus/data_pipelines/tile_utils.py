import math

#helper class
class PointF():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class PointPX():
    def __init__(self, tx, ty, px, py):
        self.tx = tx
        self.ty = ty
        self.px = px
        self.py = py

# utils
def bound(val, valMin, valMax):
    res = val if (val > valMin) else valMin #max and min are in SQL functions lib
    res = val if (res < valMax) else valMax
    return res

def degreesToRadians(deg):
    return deg * (math.pi / 180);


def radiansToDegrees(rad):
    return rad / (math.pi / 180);

class tilesUtils():
    # Consts
    TILE_SIZE = 256
    _pixelOrigin = PointF(TILE_SIZE / 2.0,TILE_SIZE / 2.0);
    _pixelsPerLonDegree = TILE_SIZE / 360.0;
    _pixelsPerLonRadian = TILE_SIZE / (2 * math.pi);

    _zoom = 1

    def __init__(self, zoom=1):
        self._zoom = zoom


    def fromLatLngToPoint(self, lat, lng, zoom=_zoom):
        point = PointF(0, 0)
        point.x = self._pixelOrigin.x + lng * self._pixelsPerLonDegree

        # Truncating to 0.9999 effectively limits latitude to 89.189. This is
        # about a third of a tile past the edge of the world tile.
        siny = bound(math.sin(degreesToRadians(lat)), -0.9999,0.9999)
        point.y = self._pixelOrigin.y + 0.5 * math.log((1 + siny) / (1 - siny)) *- self._pixelsPerLonRadian

        numTiles = 1 << zoom
        point.x = point.x * numTiles
        point.y = point.y * numTiles
        return point


    def fromPointToLatLng(self, point, zoom=_zoom):
        numTiles = 1 << zoom
        point.x = point.x / numTiles
        point.y = point.y / numTiles

        lng = (point.x - self._pixelOrigin.x) / self._pixelsPerLonDegree
        latRadians = (point.y - self._pixelOrigin.y) / - self._pixelsPerLonRadian
        lat = radiansToDegrees(2 * math.atan(math.exp(latRadians)) - math.pi / 2)
        return PointF(lat, lng)

    def fromLatLngToTile(self, lat, lng, zoom=_zoom):
        xy = self.fromLatLngToPoint(lat, lng, zoom)
        xy.x = math.floor(xy.x / self.TILE_SIZE)
        xy.y = math.floor(xy.y /self.TILE_SIZE)
        return PointF(xy.x, xy.y)



    def fromLatLngToTilePixel(self, lat, lng, zoom=_zoom):
        tile = self.fromLatLngToTile(lat, lng, zoom)

        point = self.fromLatLngToPoint(lat, lng, zoom)
        point.x = point.x % self.TILE_SIZE
        point.y = point.y % self.TILE_SIZE

        return PointPX(tile.x, tile.y, point.x, point.y)