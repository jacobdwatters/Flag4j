package com.flag4j.cvector;

import com.flag4j.CVector;
import com.flag4j.Shape;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorConstructorTests {
    int expSize;
    CNumber fillValue;
    double fillValueD;
    Shape expShape;
    CNumber[] expEntries;
    double[] entriesD;
    int[] entriesI;
    CVector a, b;

    @Test
    void sizeTest() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber();
        }

        a = new CVector(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 2 ------------
        expSize = 0;
        expShape = new Shape(new int[]{expSize});
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber();
        }

        a = new CVector(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 3 ------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new Vector(expSize));
    }

    @Test
    void sizeFillTest() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        fillValue = new CNumber(-10.23423, 100.2);
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(fillValue);
        }

        a = new CVector(expSize, fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 2 ------------
        expSize = 0;
        fillValue = new CNumber(-10.23423, 100.2);
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(fillValue);
        }

        a = new CVector(expSize,fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 3 ------------
        expSize = -1;
        fillValue = new CNumber(-10.23423, 100.2);
        assertThrows(IllegalArgumentException.class, () -> new CVector(expSize, fillValue));
    }


    @Test
    void sizeFillDTest() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        fillValueD = 10.234;
        expShape = new Shape(new int[]{expSize});
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(fillValueD);
        }

        a = new CVector(expSize, fillValueD);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 2 ------------
        expSize = 0;
        fillValueD = -10.234;
        expShape = new Shape(new int[]{expSize});
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(fillValueD);
        }

        a = new CVector(expSize,fillValueD);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 3 ------------
        expSize = -1;
        fillValueD = -10.234;
        assertThrows(IllegalArgumentException.class, () -> new CVector(expSize, fillValueD));
    }


    @Test
    void entriesTest() {
        // ----------- Sub-case 1 ------------
        entriesD = new double[]{1.0433, 2, -3, 4, 5, 6, 7, 100, -0.1231};
        expSize = entriesD.length;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesD[i]);
        }

        a = new CVector(entriesD);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 2 ------------
        entriesD = new double[]{-0.234974};
        expSize = entriesD.length;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesD[i]);
        }

        a = new CVector(entriesD);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }
    }


    @Test
    void entriesITest() {
        // ----------- Sub-case 1 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesI[i]);
        }

        a = new CVector(entriesI);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 2 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesI[i]);
        }

        a = new CVector(entriesI);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }
    }


    @Test
    void entriesCTest() {
        // ----------- Sub-case 1 ------------
        expEntries = new CNumber[]{new CNumber(100, 234.13), new CNumber(-0.992, 113.3),
                new CNumber(-0.0000000000001), new CNumber(0, -342.13)};
        expSize = expEntries.length;
        expShape = new Shape(expSize);

        a = new CVector(expEntries);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 2 ------------
        expEntries = new CNumber[]{new CNumber(-22, -0.92)};
        expSize = expEntries.length;
        expShape = new Shape(expSize);

        a = new CVector(expEntries);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }
    }


    @Test
    void copyTest() {
        // ----------- Sub-case 1 ------------
        expEntries = new CNumber[]{new CNumber(100, 234.13), new CNumber(-0.992, 113.3),
                new CNumber(-0.0000000000001), new CNumber(0, -342.13)};
        expSize = expEntries.length;
        expShape = new Shape(expSize);

        b = new CVector(expEntries);
        a = new CVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 2 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesI[i]);
        }

        b = new CVector(entriesI);
        a = new CVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 3 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesI[i]);
        }

        b = new CVector(entriesI);
        a = new CVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
                for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 4 ------------
        entriesI = new int[]{-22};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesI[i]);
        }

        b = new CVector(entriesI);
        a = new CVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }
    }

}
