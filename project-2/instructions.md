# Instruktioner

1. Behöver ett nätverk som kan prediktera ett 4-bitars XOR-mönster. HAR: ett nätverk för 2-bitars.
2. Träna nätverket tills det är bra.
- En gång blev det klockrent. Ibland är det från 0.2-1.1, där en avrundning hade gett rätt värde.
En gång var näst sista outputen fel (övriga korrekta). epochCount 10_000, learningRate 0.01. 
- FÖRSLAG: Om träningen går dåligt (kolla precisionen), kör om träningen. Återinitiera parametrarna
- i dense-lagret. 
3. Lägg till drivers för gpio.
4. Erik tipsade om for-loop för att gå igenom varenda knapptryckningssekvens och se om predikerat
resultat stämmer överens med faktiskt resultat.
5. Koppla in hårdvara (RPi Zero, knappar, LED).
