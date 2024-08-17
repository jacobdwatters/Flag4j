package org.flag4j.sparse_complex_vector;

import org.flag4j.arrays_old.sparse.CooCVector;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CooCVectorConstructorTests {
    int expSize;
    Shape expShape;
    CNumber[] expEntries;
    double[] expEntriesD;
    int[] expEntriesI;
    int[] expIndices;
    CooCVector a, b;


    @Test
    void sizeTestCase() {
        // ------------- Sub-case 1 -------------
        expSize = 114;
        expShape = new Shape(expSize);
        expEntries = new CNumber[0];
        expIndices = new int[0];
        a = new CooCVector(expSize);

        assertEquals(expSize, a.size());
        Assertions.assertEquals(expShape, a.shape);
        Assertions.assertArrayEquals(expEntries, a.entries);
        assertArrayEquals(expIndices, a.indices);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize));
    }


    @Test
    void sizeEntriesIndicesTestCase() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
            new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        a = new CooCVector(expSize, expEntries, expIndices);

        assertEquals(expSize, a.size());
        Assertions.assertEquals(expShape, a.shape);
        Assertions.assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntries, expIndices));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntries, expIndices));


        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntries, expIndices));
    }


    @Test
    void sizeEntriesIntIndicesTestCase() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesI.length];
        for(int i=0; i<expEntriesI.length; i++) {
            expEntries[i] = new CNumber(expEntriesI[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        a = new CooCVector(expSize, expEntriesI, expIndices);

        assertEquals(expSize, a.size());
        Assertions.assertEquals(expShape, a.shape);
        Assertions.assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        expIndices = new int[]{0, 11, 10003, 20034};
        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntriesI, expIndices));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesI.length];
        for(int i=0; i<expEntriesI.length; i++) {
            expEntries[i] = new CNumber(expEntriesI[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntriesI, expIndices));


        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesI.length];
        for(int i=0; i<expEntriesI.length; i++) {
            expEntries[i] = new CNumber(expEntriesI[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntriesI, expIndices));
    }


    @Test
    void sizeEntriesDoubleIndicesTestCase() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        a = new CooCVector(expSize, expEntriesD, expIndices);

        assertEquals(expSize, a.size());
        Assertions.assertEquals(expShape, a.shape);
        Assertions.assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        expIndices = new int[]{0, 11, 10003, 20034};
        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntriesD, expIndices));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntriesD, expIndices));


        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooCVector(expSize, expEntriesD, expIndices));
    }


    @Test
    void copyTestCase() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        b = new CooCVector(expSize, expEntries, expIndices);
        a = new CooCVector(b);

        assertEquals(expSize, a.size());
        Assertions.assertEquals(expShape, a.shape);
        Assertions.assertArrayEquals(expEntries, a.entries);
    }
}
