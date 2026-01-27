Filtrera genom filter som letar efter specifikt mönster

Om vi har en kernel/filter på 2x2, => 2/2 = 1 nolla runt hela bilden. Kallas padding, vaddering.

kernel/filter 2x2 kör feedforward. Varje 2x2-ruta multipliceras med vårat filter (0 0/0 1 * 0,2 0,4/0,6 0,8)

max-poolinglagret går igenom feature map och tar ut de mest framträdande attributen (rutorna med
högst värde i en 2x2-scan av feature map).


Du får ta Eriks exempel från L25, men det ska implementeras som en KLASS, inte en struct.
Structen finns i branchen 'conv'.

