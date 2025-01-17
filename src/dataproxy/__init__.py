"""

Dataaproxy maneja el acceso a datos publicos de los exchanges, adaptadas al 
formato de usode neuralab.


TODO: 
1. Conectar al los aggregate trade streams de binance y kraken
2. sincronizar los eventos para generar los vwap's a una 
   resolucion especifica

Dataproxy maneja la recepcion de streams mediante un solo pool
de websocket workers para todos los streams.




"""


class Dataproxy:
    
    def __init__(self):
        ...
        