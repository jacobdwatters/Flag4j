package com.flag4j.tensor;

import com.flag4j.*;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class TensorAddTests {
    static double[] aEntries;
    static Tensor A;
    static Shape aShape, bShape, expShape;

    int[][] sparseIndices;

    static void denseSetup() {
        aEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        aShape = new Shape(2, 3, 2);
        A = new Tensor(aShape, aEntries);
    }

    static void sparseSetup() {
        aEntries = new double[]{
                1.23, 0, 0, 0, 0, -99.23425,
                0, 2.677, 0, -0.000194, 0, 0
        };
        aShape = new Shape(2, 3, 2);
        A = new Tensor(aShape, aEntries);
    }


    @Test
    void realDenseTest() {
        denseSetup();

        double[] bEntries, expEntries;
        Tensor B, exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2);
        B = new Tensor(bShape, bEntries);
        expEntries = new double[]{
            aEntries[0]+bEntries[0], aEntries[1]+bEntries[1], aEntries[2]+bEntries[2],
            aEntries[3]+bEntries[3], aEntries[4]+bEntries[4], aEntries[5]+bEntries[5],
            aEntries[6]+bEntries[6], aEntries[7]+bEntries[7], aEntries[8]+bEntries[8],
            aEntries[9]+bEntries[9], aEntries[10]+bEntries[10], aEntries[11]+bEntries[11]
        };
        expShape = new Shape(2, 3, 2);
        exp = new Tensor(bShape, expEntries);

        assertEquals(exp, A.add(B));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2, 1);
        B = new Tensor(bShape, bEntries);

        Tensor finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.add(finalB));

        // ----------------------- Sub-case 3 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0, 1.4, 5
        };
        bShape = new Shape(7, 2);
        B = new Tensor(bShape, bEntries);

        Tensor finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.add(finalB1));
    }


    @Test
    void realSparseTest() {
        denseSetup();

        double[] bEntries, expEntries;
        SparseTensor B;
        Tensor exp;

        // TODO: create tests
    }


    @Test
    void complexDenseTest() {
        denseSetup();

        CNumber[] bEntries, expEntries;
        CTensor B, exp;

        // TODO: create tests
    }


    @Test
    void complexSparseTest() {
        denseSetup();

        CNumber[] bEntries, expEntries;
        SparseCTensor B;
        CTensor exp;

        // TODO: create tests
    }
}
