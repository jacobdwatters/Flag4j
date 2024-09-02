package org.flag4j.complex_vector;

import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.arrays.Shape;
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
    CVectorOld a, b;

    @Test
    void sizeTestCase() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = CNumber.ZERO;
        }

        a = new CVectorOld(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 2 ------------
        expSize = 0;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = CNumber.ZERO;
        }

        a = new CVectorOld(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 3 ------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new VectorOld(expSize));
    }

    @Test
    void sizeFillTestCase() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        fillValue = new CNumber(-10.23423, 100.2);
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = fillValue;
        }

        a = new CVectorOld(expSize, fillValue);

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
            expEntries[i] = fillValue;
        }

        a = new CVectorOld(expSize,fillValue);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 3 ------------
        expSize = -1;
        fillValue = new CNumber(-10.23423, 100.2);
        assertThrows(IllegalArgumentException.class, () -> new CVectorOld(expSize, fillValue));
    }


    @Test
    void sizeFillDTestCase() {
        // ----------- Sub-case 1 ------------
        expSize = 5;
        fillValueD = 10.234;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(fillValueD);
        }

        a = new CVectorOld(expSize, fillValueD);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 2 ------------
        expSize = 0;
        fillValueD = -10.234;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(fillValueD);
        }

        a = new CVectorOld(expSize,fillValueD);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 3 ------------
        expSize = -1;
        fillValueD = -10.234;
        assertThrows(IllegalArgumentException.class, () -> new CVectorOld(expSize, fillValueD));
    }


    @Test
    void entriesTestCase() {
        // ----------- Sub-case 1 ------------
        entriesD = new double[]{1.0433, 2, -3, 4, 5, 6, 7, 100, -0.1231};
        expSize = entriesD.length;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesD[i]);
        }

        a = new CVectorOld(entriesD);

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

        a = new CVectorOld(entriesD);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }
    }


    @Test
    void entriesITestCase() {
        // ----------- Sub-case 1 ------------
        entriesI = new int[]{0, 2, -3, 4, 5, 6, 7, 100, -9924};
        expSize = entriesI.length;
        expShape = new Shape(expSize);
        expEntries = new CNumber[expSize];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesI[i]);
        }

        a = new CVectorOld(entriesI);

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

        a = new CVectorOld(entriesI);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }
    }


    @Test
    void entriesCTestCase() {
        // ----------- Sub-case 1 ------------
        expEntries = new CNumber[]{new CNumber(100, 234.13), new CNumber(-0.992, 113.3),
                new CNumber(-0.0000000000001), new CNumber(0, -342.13)};
        expSize = expEntries.length;
        expShape = new Shape(expSize);

        a = new CVectorOld(expEntries);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }

        // ----------- Sub-case 2 ------------
        expEntries = new CNumber[]{new CNumber(-22, -0.92)};
        expSize = expEntries.length;
        expShape = new Shape(expSize);

        a = new CVectorOld(expEntries);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }
    }


    @Test
    void copyTestCase() {
        // ----------- Sub-case 1 ------------
        expEntries = new CNumber[]{new CNumber(100, 234.13), new CNumber(-0.992, 113.3),
                new CNumber(-0.0000000000001), new CNumber(0, -342.13)};
        expSize = expEntries.length;
        expShape = new Shape(expSize);

        b = new CVectorOld(expEntries);
        a = new CVectorOld(b);

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

        b = new CVectorOld(entriesI);
        a = new CVectorOld(b);

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

        b = new CVectorOld(entriesI);
        a = new CVectorOld(b);

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

        b = new CVectorOld(entriesI);
        a = new CVectorOld(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        for(int i=0; i<expEntries.length; i++) {
            assertEquals(expEntries[i], a.entries[i]);
        }
    }

}
