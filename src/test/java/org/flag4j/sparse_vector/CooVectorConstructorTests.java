package org.flag4j.sparse_vector;

import org.flag4j.core.Shape;
import org.flag4j.sparse.CooVector;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class CooVectorConstructorTests {
    int expSize;
    Shape expShape;
    double[] expEntries;
    int[] expEntriesI;
    int[] expIndices;
    CooVector a, b;


    @Test
    void sizeTestCase() {
        // ------------- Sub-case 1 -------------
        expSize = 114;
        expShape = new Shape(expSize);
        expEntries = new double[0];
        expIndices = new int[0];
        a = new CooVector(expSize);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);
        assertArrayEquals(expIndices, a.indices);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize));
    }


    @Test
    void sizeEntriesIndicesTestCase() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        a = new CooVector(expSize, expEntries, expIndices);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize, expEntries, expIndices));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize, expEntries, expIndices));


        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize, expEntries, expIndices));
    }


    @Test
    void sizeEntriesIntIndicesTestCase() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        a = new CooVector(expSize, expEntriesI, expIndices);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);

        // ------------- Sub-case 2 -------------
        expSize = -1;
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034};
        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize, expEntriesI, expIndices));

        // ------------- Sub-case 3 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize, expEntriesI, expIndices));


        // ------------- Sub-case 4 -------------
        expSize = 3;
        expShape = new Shape(expSize);
        expEntriesI = new int[]{1, 4, 5, 1001, -11};
        expEntries = Arrays.stream(expEntriesI).asDoubleStream().toArray();
        expIndices = new int[]{0, 11, 10003, 20034};

        assertThrows(IllegalArgumentException.class, () -> new CooVector(expSize, expEntriesI, expIndices));
    }


    @Test
    void copyTestCase() {
        // ------------- Sub-case 1 -------------
        expSize = 1001234;
        expShape = new Shape(expSize);
        expEntries = new double[]{1, 4, 5, 1001, -11.234};
        expIndices = new int[]{0, 11, 10003, 20034, 1001233};
        b = new CooVector(expSize, expEntries, expIndices);
        a = new CooVector(b);

        assertEquals(expSize, a.size());
        assertEquals(expShape, a.shape);
        assertArrayEquals(expEntries, a.entries);
    }
}
