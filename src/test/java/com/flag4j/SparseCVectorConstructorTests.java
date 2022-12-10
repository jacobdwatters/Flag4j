package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.VectorOrientations;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SparseCVectorConstructorTests {
    int expSize;
    Shape expShape;
    CNumber[] expEntries;
    double[] expEntriesD;
    int[] expEntriesI;
    int[] expIndices;
    VectorOrientations expOrientation;
    SparseCVector a, b;


    @Test
    void sizeTest() {
        // ------------- Sub-case 1 -------------
        expSize = 114;
        expShape = new Shape(expSize);
        expEntries = new CNumber[0];
        expIndices = new int[0];
        expOrientation = VectorOrientations.COL;
        a = new SparseCVector(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);
        assertArrayEquals(expIndices, a.indices);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new SparseCVector(expSize));
    }


    @Test
    void sizeEntriesIndicesTest() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
            new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientations.COL;
        a = new SparseCVector(expSize, expEntries, expIndices);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new SparseCVector(expSize, expEntries, expIndices));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new SparseCVector(expSize, expEntries, expIndices));


        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new SparseCVector(expSize, expEntries, expIndices));
    }


    @Test
    void sizeEntriesIntIndicesTest() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesI.length];
        for(int i=0; i<expEntriesI.length; i++) {
            expEntries[i] = new CNumber(expEntriesI[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientations.COL;
        a = new SparseCVector(expSize, expEntriesI, expIndices);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        expIndices = new int[]{0, 11, 10003, 20034};
        assertThrows(IllegalArgumentException.class, () -> new SparseCVector(expSize, expEntriesI, expIndices));

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

        assertThrows(IllegalArgumentException.class, () -> new SparseCVector(expSize, expEntriesI, expIndices));


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

        assertThrows(IllegalArgumentException.class, () -> new SparseCVector(expSize, expEntriesI, expIndices));
    }


    @Test
    void sizeEntriesDoubleIndicesTest() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientations.COL;
        a = new SparseCVector(expSize, expEntriesD, expIndices);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        expIndices = new int[]{0, 11, 10003, 20034};
        assertThrows(IllegalArgumentException.class, () -> new SparseCVector(expSize, expEntriesD, expIndices));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new SparseCVector(expSize, expEntriesD, expIndices));


        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new SparseCVector(expSize, expEntriesD, expIndices));
    }


    @Test
    void sizeOrientationTest() {
        // ------------- Sub-case 1 -------------
        expSize = 114;
        expShape = new Shape(expSize);
        expEntries = new CNumber[0];
        expIndices = new int[0];
        expOrientation = VectorOrientations.COL;
        a = new SparseCVector(expSize, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);
        assertArrayEquals(expIndices, a.indices);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new SparseCVector(expSize, expOrientation));


        // ------------- Sub-case 3 -------------
        expSize = 114;
        expShape = new Shape(expSize);
        expEntries = new CNumber[0];
        expIndices = new int[0];
        expOrientation = VectorOrientations.ROW;
        a = new SparseCVector(expSize, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);
        assertArrayEquals(expIndices, a.indices);
    }


    @Test
    void sizeOrientationEntriesIndicesTest() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientations.COL;
        a = new SparseCVector(expSize, expEntries, expIndices, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        expOrientation = VectorOrientations.COL;
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034};
        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntries, expIndices, expOrientation));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntries, expIndices, expOrientation));

        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntries, expIndices, expOrientation));

        // ------------- Sub-case 5 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientations.ROW;
        a = new SparseCVector(expSize, expEntries, expIndices, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 6 -------------
        expSize = -1;
        expOrientation = VectorOrientations.ROW;
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034};
        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntries, expIndices, expOrientation));

        // ------------- Sub-case 7 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expOrientation = VectorOrientations.ROW;
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntries, expIndices, expOrientation));

        // ------------- Sub-case 8 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expOrientation = VectorOrientations.ROW;
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntries, expIndices, expOrientation));
    }


    @Test
    void sizeOrientationEntriesIntIndicesTest() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesI.length];
        for(int i=0; i<expEntriesI.length; i++) {
            expEntries[i] = new CNumber(expEntriesI[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientations.COL;
        a = new SparseCVector(expSize, expEntriesI, expIndices, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        expOrientation = VectorOrientations.COL;
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesI.length];
        for(int i=0; i<expEntriesI.length; i++) {
            expEntries[i] = new CNumber(expEntriesI[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntriesI, expIndices, expOrientation));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expOrientation = VectorOrientations.COL;
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesI.length];
        for(int i=0; i<expEntriesI.length; i++) {
            expEntries[i] = new CNumber(expEntriesI[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntriesI, expIndices, expOrientation));

        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expOrientation = VectorOrientations.COL;
        expEntries = new CNumber[expEntriesI.length];
        for(int i=0; i<expEntriesI.length; i++) {
            expEntries[i] = new CNumber(expEntriesI[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntriesI, expIndices, expOrientation));

        // ------------- Sub-case 5 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesI.length];
        for(int i=0; i<expEntriesI.length; i++) {
            expEntries[i] = new CNumber(expEntriesI[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientations.ROW;
        a = new SparseCVector(expSize, expEntriesI, expIndices, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 6 -------------
        expSize = -1;
        expOrientation = VectorOrientations.ROW;
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesI.length];
        for(int i=0; i<expEntriesI.length; i++) {
            expEntries[i] = new CNumber(expEntriesI[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntriesI, expIndices, expOrientation));

        // ------------- Sub-case 7 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expOrientation = VectorOrientations.ROW;
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesI.length];
        for(int i=0; i<expEntriesI.length; i++) {
            expEntries[i] = new CNumber(expEntriesI[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntriesI, expIndices, expOrientation));

        // ------------- Sub-case 8 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expOrientation = VectorOrientations.ROW;
        expEntries = new CNumber[expEntriesI.length];
        for(int i=0; i<expEntriesI.length; i++) {
            expEntries[i] = new CNumber(expEntriesI[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntriesI, expIndices, expOrientation));
    }


    @Test
    void sizeOrientationEntriesDoubleIndicesTest() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientations.COL;
        a = new SparseCVector(expSize, expEntriesD, expIndices, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        expOrientation = VectorOrientations.COL;
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntriesD, expIndices, expOrientation));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expOrientation = VectorOrientations.COL;
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntriesD, expIndices, expOrientation));

        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expOrientation = VectorOrientations.COL;
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntriesD, expIndices, expOrientation));

        // ------------- Sub-case 5 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientations.ROW;
        a = new SparseCVector(expSize, expEntriesD, expIndices, expOrientation);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 6 -------------
        expSize = -1;
        expOrientation = VectorOrientations.ROW;
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntriesD, expIndices, expOrientation));

        // ------------- Sub-case 7 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expOrientation = VectorOrientations.ROW;
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntriesD, expIndices, expOrientation));

        // ------------- Sub-case 8 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntriesD = new double[]{1, 4, 5, 1001, -11};
        expOrientation = VectorOrientations.ROW;
        expEntries = new CNumber[expEntriesD.length];
        for(int i=0; i<expEntriesD.length; i++) {
            expEntries[i] = new CNumber(expEntriesD[i]);
        }
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class,
                () -> new SparseCVector(expSize, expEntriesD, expIndices, expOrientation));
    }


    @Test
    void copyTest() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientations.COL;
        b = new SparseCVector(expSize, expEntries, expIndices, expOrientation);
        a = new SparseCVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 5 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new CNumber[]{new CNumber(-1233, 1.3314), new CNumber(9034, 10.23445),
                new CNumber(83.133, -334), new CNumber(-92.133, -9.4), new CNumber(0, 13.435)};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        expOrientation = VectorOrientations.ROW;
        b = new SparseCVector(expSize, expEntries, expIndices, expOrientation);
        a = new SparseCVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertEquals(expOrientation, a.getOrientation());
        assertArrayEquals(expEntries, a.entries);
    }
}
