package com.flag4j.complex_matrix;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.Shape;
import com.flag4j.dense.CMatrix;
import com.flag4j.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CMatrixConstructorTests {
    CNumber[][] entries;
    String[][] entriesS;
    double[][] entriesD;
    int[][] entriesI;
    CNumber[] expEntries, entriesC1D;
    int expNumRows;
    int expNumCols;
    int size;
    Shape shape;
    CNumber value;
    double valueD;

    CMatrix A, B;
    Matrix C;


    @Test
    void sizeConstructorTestCase() {
        // -------------- Sub-case 1 --------------
        size = 5;
        expEntries = new CNumber[size*size];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber();
        }

        expNumRows = size;
        expNumCols = size;
        A = new CMatrix(size);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- Sub-case 2 --------------
        size = 2;
        expEntries = new CNumber[size*size];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber();
        }

        expNumRows = size;
        expNumCols = size;
        A = new CMatrix(size);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);

        // -------------- Sub-case 3 --------------
        size = -2;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(size));
    }


    @Test
    void valueSizeTestCase() {
        // -------------- Sub-case 1 --------------
        size = 5;
        value = new CNumber(-23.13, 100.442);
        expEntries = new CNumber[size*size];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = value.copy();
        }

        expNumRows = size;
        expNumCols = size;
        A = new CMatrix(size, value);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- Sub-case 2 --------------
        size = 5;
        value = new CNumber(0, 5);
        expEntries = new CNumber[size*size];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = value.copy();
        }

        expNumRows = size;
        expNumCols = size;
        A = new CMatrix(size, value);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- Sub-case 3 --------------
        value = new CNumber(0, 5);
        size = -2;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(size, value));
    }


    @Test
    void valueSizeDoubleTestCase() {
        // -------------- Sub-case 1 --------------
        size = 5;
        valueD = 9.1;
        expEntries = new CNumber[size*size];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(valueD);
        }

        expNumRows = size;
        expNumCols = size;
        A = new CMatrix(size, valueD);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- Sub-case 2 --------------
        size = 5;
        valueD = -0.000013;
        expEntries = new CNumber[size*size];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(valueD);
        }

        expNumRows = size;
        expNumCols = size;
        A = new CMatrix(size, valueD);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);

        // -------------- Sub-case 3 --------------
        valueD = 5;
        size = -100;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(size, valueD));
    }


    @Test
    void rowsColsTestCase() {
        // -------------- Sub-case 1 --------------
        expNumRows = 5;
        expNumCols = 2;
        expEntries = new CNumber[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber();
        }

        A = new CMatrix(expNumRows, expNumCols);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- Sub-case 2 --------------
        expNumRows = 1;
        expNumCols = 16;
        expEntries = new CNumber[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber();
        }

        A = new CMatrix(expNumRows, expNumCols);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- Sub-case 3 --------------
        expNumRows = -1;
        expNumCols = 2;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols));

        // -------------- Sub-case 4 --------------
        expNumRows = 1;
        expNumCols = -2;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols));

        // -------------- Sub-case 5 --------------
        expNumRows = -1;
        expNumCols = -2;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols));
    }

    @Test
    void rowColValueTestCase() {
        // -------------- Sub-case 1 --------------
        expNumRows = 5;
        expNumCols = 2;
        expEntries = new CNumber[expNumRows*expNumCols];
        value = new CNumber(900.13);

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = value.copy();
        }

        A = new CMatrix(expNumRows, expNumCols, value);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- Sub-case 2 --------------
        expNumRows = 1;
        expNumCols = 16;
        expEntries = new CNumber[expNumRows*expNumCols];
        value = new CNumber(-1, -13);

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = value.copy();
        }

        A = new CMatrix(expNumRows, expNumCols, value);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- Sub-case 3 --------------
        expNumRows = -1;
        expNumCols = 2;
        value = new CNumber(1, 1);

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols, value));

        // -------------- Sub-case 4 --------------
        expNumRows = 1;
        expNumCols = -2;
        value = new CNumber(1, 1);

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols, value));

        // -------------- Sub-case 5 --------------
        expNumRows = -1;
        expNumCols = -2;
        value = new CNumber(1, 1);

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols, value));
    }

    @Test
    void rowColValueDoubleTestCase() {
        // -------------- Sub-case 1 --------------
        expNumRows = 5;
        expNumCols = 2;
        expEntries = new CNumber[expNumRows*expNumCols];
        valueD = 14;

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(valueD);
        }

        A = new CMatrix(expNumRows, expNumCols, valueD);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);

        // -------------- Sub-case 2 --------------
        expNumRows = 1;
        expNumCols = 16;
        expEntries = new CNumber[expNumRows*expNumCols];
        valueD = Double.POSITIVE_INFINITY;

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(valueD);
        }

        A = new CMatrix(expNumRows, expNumCols, valueD);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);

        // -------------- Sub-case 3 --------------
        expNumRows = -1;
        expNumCols = 2;
        valueD = 1;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols, valueD));

        // -------------- Sub-case 4 --------------
        expNumRows = 1;
        expNumCols = -2;
        valueD = 1;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols, valueD));

        // -------------- Sub-case 5 --------------
        expNumRows = -1;
        expNumCols = -2;
        valueD = 1;

        assertThrows(IllegalArgumentException.class, () -> new CMatrix(expNumRows, expNumCols, valueD));
    }


    @Test
    void shapeTestCase() {
        // -------------- Sub-case 1 --------------
        expNumRows = 5;
        expNumCols = 2;
        shape = new Shape(expNumRows, expNumCols);
        expEntries = new CNumber[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber();
        }

        A = new CMatrix(shape);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- Sub-case 2 --------------
        expNumRows = 1;
        expNumCols = 16;
        shape = new Shape(expNumRows, expNumCols);
        expEntries = new CNumber[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber();
        }

        A = new CMatrix(shape);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }


    @Test
    void shapeValueTestCase() {
        // -------------- Sub-case 1 --------------
        expNumRows = 5;
        expNumCols = 2;
        shape = new Shape(expNumRows, expNumCols);
        value = new CNumber(19, -1345.0001);
        expEntries = new CNumber[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = value.copy();
        }

        A = new CMatrix(shape, value);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);


        // -------------- Sub-case 2 --------------
        expNumRows = 1;
        expNumCols = 16;
        shape = new Shape(expNumRows, expNumCols);
        value = new CNumber(1.3, 1.3566);
        expEntries = new CNumber[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = value.copy();
        }

        A = new CMatrix(shape, value);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }


    @Test
    void shapeValueDoubleTestCase() {
        // -------------- Sub-case 1 --------------
        expNumRows = 5;
        expNumCols = 2;
        shape = new Shape(expNumRows, expNumCols);
        valueD = 13.3456;
        expEntries = new CNumber[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(valueD);
        }

        A = new CMatrix(shape, valueD);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);

        // -------------- Sub-case 2 --------------
        expNumRows = 1;
        expNumCols = 16;
        shape = new Shape(expNumRows, expNumCols);
        valueD = -99.3137;
        expEntries = new CNumber[expNumRows*expNumCols];

        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(valueD);
        }

        A = new CMatrix(shape, valueD);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }

    @Test
    void arrTestCase() {
        entries = new CNumber[][]{{new CNumber("100.234-0.0103i"), new CNumber("134.5")},
                {new CNumber("i"), new CNumber("100.3465i")},
                {new CNumber("-0.9344-1.345i"), new CNumber("-103894.1334")},
                {new CNumber("0"), new CNumber(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY)}};
        expNumRows = 4;
        expNumCols = 2;
        expEntries = new CNumber[expNumRows*expNumCols];

        int count=0;
        for(int i=0; i< entries.length; i++) {
            for(int j=0; j<entries[0].length; j++) {
                expEntries[count++] = new CNumber(entries[i][j]);
            }
        }
        A = new CMatrix(entries);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }

    @Test
    void arrStringTestCase() {
        entriesS = new String[][]{{"1", "-2+8.133i", "0.13334-i"}, {"133.4", "-29.13i", "8+9i"}};
        expNumRows = 2;
        expNumCols = 3;
        expEntries = new CNumber[expNumRows*expNumCols];

        int count=0;
        for(int i=0; i< entriesS.length; i++) {
            for(int j=0; j<entriesS[0].length; j++) {
                expEntries[count++] = new CNumber(entriesS[i][j]);
            }
        }
        A = new CMatrix(entriesS);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }

    @Test
    void arrDoubleTestCase() {
        entriesD = new double[][]{{1, -2, 0.13334, 15.3, Double.NEGATIVE_INFINITY},
                {133.4, -29.13, 8, 0, Double.POSITIVE_INFINITY}};
        expNumRows = 2;
        expNumCols = 5;
        expEntries = new CNumber[expNumRows*expNumCols];

        int count=0;
        for(int i=0; i< entriesD.length; i++) {
            for(int j=0; j<entriesD[0].length; j++) {
                expEntries[count++] = new CNumber(entriesD[i][j]);
            }
        }
        A = new CMatrix(entriesD);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }


    @Test
    void arrIntTestCase() {
        entriesI = new int[][]{{1, -2, 0, 15, 100},
                {133, -29, 8, 0, -1000}};
        expNumRows = 2;
        expNumCols = 5;
        expEntries = new CNumber[expNumRows*expNumCols];

        int count=0;
        for(int i=0; i< entriesI.length; i++) {
            for(int j=0; j<entriesI[0].length; j++) {
                expEntries[count++] = new CNumber(entriesI[i][j]);
            }
        }
        A = new CMatrix(entriesI);

        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }


    @Test
    void matTestCase() {
        entriesD = new double[][]{{1, -2, 0.13334, 15.3, Double.NEGATIVE_INFINITY},
                {133.4, -29.13, 8, 0, Double.POSITIVE_INFINITY}};
        C = new Matrix(entriesD);
        expNumRows = 2;
        expNumCols = 5;
        expEntries = new CNumber[expNumRows*expNumCols];

        int count=0;
        for(int i=0; i< entriesD.length; i++) {
            for(int j=0; j<entriesD[0].length; j++) {
                expEntries[count++] = new CNumber(entriesD[i][j]);
            }
        }

        A = new CMatrix(C);
        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }


    @Test
    void cmatTestCase() {
        entriesS = new String[][]{{"1", "-2+8.133i", "0.13334-i"}, {"133.4", "-29.13i", "8+9i"}};
        B = new CMatrix(entriesS);
        expNumRows = 2;
        expNumCols = 3;
        expEntries = new CNumber[expNumRows*expNumCols];

        int count=0;
        for(int i=0; i< entriesS.length; i++) {
            for(int j=0; j<entriesS[0].length; j++) {
                expEntries[count++] = new CNumber(entriesS[i][j]);
            }
        }

        A = new CMatrix(B);
        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }


    @Test
    void rowsColsComplexTestCase() {
        entriesC1D = new CNumber[]{new CNumber(1, 345.6), new CNumber(-3441, 0.0094343431),
                new CNumber(-6, -9.2), new CNumber(Double.NEGATIVE_INFINITY, 1),
                new CNumber(9.234, -07643.2), new CNumber(6, 7),
                new CNumber(0.0000000002134), new CNumber(0, -92.2),
                new CNumber(5, 666666.4545), new CNumber(438905, 13)};
        expNumRows = 5;
        expNumCols = 2;
        expEntries = new CNumber[]{new CNumber(1, 345.6), new CNumber(-3441, 0.0094343431),
                new CNumber(-6, -9.2), new CNumber(Double.NEGATIVE_INFINITY, 1),
                new CNumber(9.234, -07643.2), new CNumber(6, 7),
                new CNumber(0.0000000002134), new CNumber(0, -92.2),
                new CNumber(5, 666666.4545), new CNumber(438905, 13)};

        A = new CMatrix(expNumRows, expNumCols, entriesC1D);
        assertArrayEquals(expEntries, A.entries);
        assertEquals(expNumRows, A.numRows);
        assertEquals(expNumCols, A.numCols);
    }
}
