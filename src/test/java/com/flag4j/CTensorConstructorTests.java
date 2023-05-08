package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class CTensorConstructorTests {

    double[] entriesD;
    int[] entriesI;
    double value;
    CNumber valueC;
    CNumber[] expEntries;
    int expRank;
    Shape expShape;
    Tensor C;
    CTensor A, B;


    @Test
    void shapeConstructorTest() {
        // -------------- Sub-case 1 --------------
        expShape = new Shape(4, 5, 6, 7, 1, 2, 4);
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber();
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // -------------- Sub-case 2 --------------
        expShape = new Shape();
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber();
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // -------------- Sub-case 3 --------------
        expShape = new Shape(1003);
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber();
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);
    }


    @Test
    void shapeValueConstructorTest() {
        // -------------- Sub-case 1 --------------
        value = 3.1345;
        expShape = new Shape(4, 5, 6, 7, 1, 2, 4);
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(value);
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape, value);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // -------------- Sub-case 2 --------------
        value = 11.4;
        expShape = new Shape();
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(value);
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape, value);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // -------------- Sub-case 3 --------------
        value = 8;
        expShape = new Shape(1003);
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(value);
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape, value);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);
    }


    @Test
    void shapeCValueConstructorTest() {
        // -------------- Sub-case 1 --------------
        valueC = new CNumber(3.1345);
        expShape = new Shape(4, 5, 6, 7, 1, 2, 4);
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(valueC);
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape, valueC);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // -------------- Sub-case 2 --------------
        valueC = new CNumber(11.4, -0.3313);
        expShape = new Shape();
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(valueC);
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape, valueC);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // -------------- Sub-case 3 --------------
        valueC = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expShape = new Shape(1003);
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(valueC);
        }
        expRank = expShape.getRank();
        A = new CTensor(expShape, valueC);
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
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesD[i]);
        }
        A = new CTensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 2 ----------
        expShape = new Shape(3, 2, 5);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesD[i]);
        }
        A = new CTensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 3 ----------
        expShape = new Shape(3, 10);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesD[i]);
        }
        A = new CTensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 4 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesD[i]);
        }
        A = new CTensor(expShape, entriesD);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 5 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, };
        expRank = expShape.getRank();

        assertThrows(IllegalArgumentException.class, () -> new CTensor(expShape, entriesD));


        // ---------- Sub-case 6 ----------
        expShape = new Shape(5, 6);
        entriesD = new double[]{1, 2, 3, 4, 5, 6, 7, -221.233, 11.33, 11,
                2, -11334, 11.33434, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.14159, 4, 8, 100, 2343,
                9.33, 3244, 156.224, 3445, 60.3, 44, 13, 1, 4, 56, 113, 34, 5};
        expRank = expShape.getRank();

        assertThrows(IllegalArgumentException.class, () -> new CTensor(expShape, entriesD));
    }


    @Test
    void shapeEntriesInt() {
        // ---------- Sub-case 1 ----------
        expShape = new Shape(1, 3, 2, 5);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesI[i]);
        }
        A = new CTensor(expShape, entriesI);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 2 ----------
        expShape = new Shape(3, 2, 5);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesI[i]);
        }
        A = new CTensor(expShape, entriesI);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 3 ----------
        expShape = new Shape(3, 10);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesI[i]);
        }
        A = new CTensor(expShape, entriesI);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 4 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1};
        expRank = expShape.getRank();
        expEntries = new CNumber[expShape.totalEntries().intValue()];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesI[i]);
        }
        A = new CTensor(expShape, entriesI);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 5 ----------
        expShape = new Shape(6, 1, 1, 1, 5, 1, 1, 1, 1, 1);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44};
        expRank = expShape.getRank();

        assertThrows(IllegalArgumentException.class, () -> new CTensor(expShape, entriesI));

        // ---------- Sub-case 6 ----------
        expShape = new Shape(5, 6);
        entriesI = new int[]{1, 2, 3, 4, 5, 6, 7, -221, 11, 11,
                2, -11334, 11, 0, 0, 3, 4, 8, 100, 2343,
                9, 3244, 156, 3445, 60, 44, 13, 908, 4, 1, 19, 2313, 112, 3};
        expRank = expShape.getRank();

        assertThrows(IllegalArgumentException.class, () -> new CTensor(expShape, entriesI));
    }


    @Test
    void shapeEntries() {
        // ---------- Sub-case 1 ----------
        expShape = new Shape(1, 3, 2, 2);
        expEntries = new CNumber[] {
                new CNumber(1, 0.32), new CNumber(11.2334, -94.45), new CNumber(94), new CNumber(0, 445.2),
                new CNumber(1, 3), new CNumber(-9, -13.4), new CNumber(34.5, 1), new CNumber(0,-1),
                new CNumber(0.0000043), new CNumber(0, 0.99810382193), new CNumber(113334, -84334), new CNumber(190, 334.55)};
        expRank = expShape.getRank();
        A = new CTensor(expShape, expEntries);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 2 ----------
        expShape = new Shape(1, 3, 2, 2);
        expEntries = new CNumber[] {
                new CNumber(1, 0.32), new CNumber(11.2334, -94.45), new CNumber(94), new CNumber(0, 445.2),
                new CNumber(1, 3), new CNumber(-9, -13.4), new CNumber(34.5, 1), new CNumber(0,-1),
                new CNumber(0.0000043), new CNumber(0, 0.99810382193), new CNumber(113334, -84334), new CNumber(190, 334.55)};
        expRank = expShape.getRank();
        A = new CTensor(expShape, expEntries);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 3 ----------
        expShape = new Shape(1, 3, 2, 2);
        expEntries = new CNumber[] {
                new CNumber(1, 0.32), new CNumber(11.2334, -94.45), new CNumber(94), new CNumber(0, 445.2),
                new CNumber(1, 3), new CNumber(-9, -13.4), new CNumber(34.5, 1), new CNumber(0,-1),
                new CNumber(0.0000043), new CNumber(0, 0.99810382193), new CNumber(113334, -84334), new CNumber(190, 334.55)};
        expRank = expShape.getRank();
        A = new CTensor(expShape, expEntries);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 4 ----------
        expShape = new Shape(1, 3, 2, 2);
        expEntries = new CNumber[] {
                new CNumber(1, 0.32), new CNumber(11.2334, -94.45), new CNumber(94), new CNumber(0, 445.2),
                new CNumber(1, 3), new CNumber(-9, -13.4), new CNumber(34.5, 1), new CNumber(0,-1),
                new CNumber(0.0000043), new CNumber(0, 0.99810382193), new CNumber(113334, -84334), new CNumber(190, 334.55)};
        expRank = expShape.getRank();
        A = new CTensor(expShape, expEntries);

        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);

        // ---------- Sub-case 5 ----------
        expShape = new Shape(1, 3, 2, 2);
        expEntries = new CNumber[] {
                new CNumber(1, 0.32), new CNumber(11.2334, -94.45), new CNumber(94), new CNumber(0, 445.2),
                new CNumber(1, 3), new CNumber(-9, -13.4), new CNumber(34.5, 1), new CNumber(0,-1),
                new CNumber(0.0000043), new CNumber(0, 0.99810382193)};
        expRank = expShape.getRank();

        assertThrows(IllegalArgumentException.class, () -> new CTensor(expShape, expEntries));

        // ---------- Sub-case 6 ----------
        expShape = new Shape(1, 3, 2, 2);
        expEntries = new CNumber[] {
                new CNumber(1, 0.32), new CNumber(11.2334, -94.45), new CNumber(94), new CNumber(0, 445.2),
                new CNumber(1, 3), new CNumber(-9, -13.4), new CNumber(34.5, 1), new CNumber(0,-1),
                new CNumber(0.0000043), new CNumber(0, 0.99810382193), new CNumber(113334, -84334), new CNumber(190, 334.55),
                new CNumber(0, 0)};
        expRank = expShape.getRank();
        assertThrows(IllegalArgumentException.class, () -> new CTensor(expShape, expEntries));
    }


    @Test
    void cTensorConstructorTest() {
        expShape = new Shape(2, 3, 1, 2);
        expEntries = new CNumber[] {
                new CNumber(1, 0.32), new CNumber(11.2334, -94.45), new CNumber(94), new CNumber(0, 445.2),
                new CNumber(1, 3), new CNumber(-9, -13.4), new CNumber(34.5, 1), new CNumber(0,-1),
                new CNumber(0.0000043), new CNumber(0, 0.99810382193), new CNumber(113334, -84334), new CNumber(190, 334.55)};
        expRank = expShape.getRank();
        B = new CTensor(expShape, expEntries);
        A = new CTensor(B);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);
    }


    @Test
    void tensorConstructorTest() {
        expShape = new Shape(2, 3, 1, 2);
        entriesD = new double[]{1, -1.4133, 113.4, 0.4, 11.3, 445, 133.445, 9.8, 13384, -993.44, 11, 12};
        expEntries = new CNumber[entriesD.length];
        for(int i=0; i<expEntries.length; i++) {
            expEntries[i] = new CNumber(entriesD[i]);
        }
        expRank = expShape.getRank();
        C = new Tensor(expShape, entriesD);
        A = new CTensor(C);
        assertEquals(expShape, A.shape);
        assertEquals(expRank, A.getRank());
        assertArrayEquals(expEntries, A.entries);
    }
}
