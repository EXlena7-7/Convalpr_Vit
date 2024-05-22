
# obj conf:
conf_camara=[{
  'camara1':{
     'roi':{
        'x' : 1000,
        'y' : 600,
        'w' : 700,
        'h' : 300 ,

     },
      'roi2':{
        'x' : 0,
        'y' : 0,
        'w' : 0,
        'h' : 0 ,

     },
     'poligono':{
          'px1':378,
          'py1':668,
          'px2':300,
          'py2':258,
          'px3':450,
          'py3':240,
          'px4':890,
          'py4':620,
          

     },

     'lineas':[
      {
         'linea':{
            'cy1':880,
            'cx1':1100,
            'cx2':1900,
            'lcolor1':255,
            'lcolor2':255,
            'lcolor3':255,
            'boder':10,
            'tx1':1050,
            'tx2':860,
            'sizetx':0.8,
            'grosor':2,
            'etiqueta':'conteo caracas',
            'txcolor1':0,
            'txcolor2':255,
            'txcolor3':255,
           
         }
      },      {
         'linea':{
            'cy1':369,
            'cx1':177,
            'cx2':927,
            'lcolor1':255,
            'lcolor2':255,
            'lcolor3':255,
            'boder':1,
            'tx1':182,
            'tx2':367,
            'sizetx':0.8,
            'grosor':2,
            'etiqueta':'L22',
            'txcolor1':0,
            'txcolor2':255,
            'txcolor3':255,


         }
      }

   ],
   'conte_vehiculos':0,
   'conte_personas':0,
   },
    'camara2':{
    
     'poligono':{
          'px1':1050,
          'py1':1000,
          'px2':1000,
          'py2':658,
          'px3':1510,
          'py3':650,
          'px4':1920,
          'py4':900,
          

     },

     'lineas':[
      {
         'linea':[1020,700,1720,700],
         
      },      {
         
         'linea':[1020,850,1820,850]
            
         
      }

   ],
    'id_deteccion':[190,634],
    'array_rojo':[190,734],
    'array_azul':[190,834],
    'array_verde':[190,934],
    'placa':[190,1000,],


   'conte_vehiculos':0,
   'conte_personas':0,
   },
   # conf camara ccaracas
       'camara3':{
    
     'poligono':{
          'px1':350,
          'py1':700,
          'px2':300,
          'py2':200,
          'px3':480,
          'py3':250,
          'px4':1250,
          'py4':700,
          

     },

    'lineas':[
      {
         'linea':[250,500,900,500],
         
      },      {
         
         'linea':[250,600,1100,600]
            
         
      }
      # camara3
   ],'id_deteccion':[90,334],
    'array_rojo':[90,360],
    'array_azul':[90,390],
    'array_verde':[90,430],
    'placa':[90,460,],
   'conte_vehiculos':0,
   'conte_personas':0,
   },


   # 
    # conf camara punto fijo 62
       'camara4':{
    
     'poligono':{
          'px1':500,
          'py1':1100,
          'px2':500,
          'py2':200,
          'px3':1800,
          'py3':200,
          'px4':1800,
          'py4':1100,
          

     },

    'lineas':[
      {
         'linea':[500,500,1800,500],
         
      },      {
         
         'linea':[500,850,1800,850]
            
         
      }
      # camara3
   ],'id_deteccion':[90,334],
    'array_rojo':[90,360],
    'array_azul':[90,390],
    'array_verde':[90,430],
    'placa':[90,460,],
   'conte_vehiculos':0,
   'conte_personas':0,
   },
},
]