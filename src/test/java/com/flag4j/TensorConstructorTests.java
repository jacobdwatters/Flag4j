package com.flag4j;

import java.util.Arrays;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TensorConstructorTests {
    double[] entriesD;
    int[] entriesI;
    double value;
    double[] expEntries;
    int expRank;
    Shape expShape;
    Tensor A, B;


    @Test
    void shapeConstructorTest() {
        // -------------- Sub-case 1 --------------
        expShape = new Shape(4, 5, 6, 7, 1, 2, 4);
        expEntries = new double[expShape.totalEntries().intValue()];
        expRank = expShape.getRank();
        A = new Tensor(expShape);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // -------------- Sub-case 2 --------------
        expShape = new Shape();
        expEntries = new double[expShape.totalEntries().intValue()];
        expRank = expShape.getRank();
        A = new Tensor(expShape);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // -------------- Sub-case 3 --------------
        expShape = new Shape(1003);
        expEntries = new double[expShape.totalEntries().intValue()];
        expRank = expShape.getRank();
        A = new Tensor(expShape);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);
    }


    @Test
    void shapeValueConstructorTest() {
        // -------------- Sub-case 1 --------------
        value = 3.1345;
        expShape = new Shape(4, 5, 6, 7, 1, 2, 4);
        expEntries = new double[expShape.totalEntries().intValue()];
        expRank = expShape.getRank();
        Arrays.fill(expEntries, value);
        A = new Tensor(expShape, value);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // -------------- Sub-case 2 --------------
        value = 11.4;
        expShape = new Shape();
        expEntries = new double[expShape.totalEntries().intValue()];
        expRank = expShape.getRank();
        Arrays.fill(expEntries, value);
        A = new Tensor(expShape, value);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // -------------- Sub-case 3 --------------
        value = 8;
        expShape = new Shape(1003);
        expEntries = new double[expShape.totalEntries().intValue()];
        expRank = expShape.getRank();
        Arrays.fill(expEntries, value);
        A = new Tensor(expShape, value);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);
    }


    @Test
    void shapeEntriesDouble() {
        // ---------- Sub-case 1 ----------
        expShape = new Shape(1, 3, 2, 5);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = entriesD.clone();
        A = new Tensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 2 ----------
        expShape = new Shape(3, 2, 5);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = entriesD.clone();
        A = new Tensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 3 ----------
        expShape = new Shape(3, 10);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = entriesD.clone();
        A = new Tensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 4 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = entriesD.clone();
        A = new Tensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 5 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, };
        expRank = expShape.getRank();
        expEntries = entriesD.clone();

        assertThrows(IllegalArgumentException.class, () -> new Tensor(expShape, entriesD));


        // ---------- Sub-case 6 ----------
        expShape = new Shape(5, 6);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 1, 4, 56, 113, 34, 5};
        expRank = expShape.getRank();
        expEntries = entriesD.clone();
        assertThrows(IllegalArgumentException.class, () -> new Tensor(expShape, entriesD));
    }


    @Test
    void shapeEntriesInt() {
        // ---------- Sub-case 1 ----------
        expShape = new Shape(1, 3, 2, 5);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = Arrays.stream(entriesI).asDoubleStream().toArray();
        A = new Tensor(expShape, entriesI);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 2 ----------
        expShape = new Shape(3, 2, 5);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = Arrays.stream(entriesI).asDoubleStream().toArray();
        A = new Tensor(expShape, entriesI);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 3 ----------
        expShape = new Shape(3, 10);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = Arrays.stream(entriesI).asDoubleStream().toArray();
        A = new Tensor(expShape, entriesI);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 4 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = Arrays.stream(entriesI).asDoubleStream().toArray();
        A = new Tensor(expShape, entriesI);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 5 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44};
        expRank = expShape.getRank();
        expEntries = Arrays.stream(entriesI).asDoubleStream().toArray();

        assertThrows(IllegalArgumentException.class, () -> new Tensor(expShape, entriesI));

        // ---------- Sub-case 6 ----------
        expShape = new Shape(5, 6);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1, 19, 2313, 112, 3};
        expRank = expShape.getRank();
        expEntries = Arrays.stream(entriesI).asDoubleStream().toArray();
        assertThrows(IllegalArgumentException.class, () -> new Tensor(expShape, entriesI));
    }


    @Test
    void tensorConstructorTest() {
        expShape = new Shape(2, 3, 1, 2);
        expEntries = new double[]{1, -1.4133, 113.4, 0.4, 11.3, 445, 133.445, 9.8, 13384, -993.44, 11, 12};
        expRank = expShape.getRank();
        B = new Tensor(expShape, expEntries);
        A = new Tensor(B);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);
    }
}
