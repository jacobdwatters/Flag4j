package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorElemDivTests {
    static double[] aEntries;
    static Tensor A;
    static Shape aShape, bShape, expShape;

    @BeforeEach
    void setup() {
        aEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        aShape = new Shape(2, 3, 2);
        A = new Tensor(aShape, aEntries);
    }


    @Test
    void realDenseTest() {
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
                aEntries[0]/bEntries[0], aEntries[1]/bEntries[1], aEntries[2]/bEntries[2],
                aEntries[3]/bEntries[3], aEntries[4]/bEntries[4], aEntries[5]/bEntries[5],
                aEntries[6]/bEntries[6], aEntries[7]/bEntries[7], aEntries[8]/bEntries[8],
                aEntries[9]/bEntries[9], aEntries[10]/bEntries[10], aEntries[11]/bEntries[11]
        };
        expShape = new Shape(2, 3, 2);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.elemDiv(B));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2, 1);
        B = new Tensor(bShape, bEntries);

        Tensor finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.elemDiv(finalB));

        // ----------------------- Sub-case 3 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0, 1.4, 5
        };
        bShape = new Shape(7, 2);
        B = new Tensor(bShape, bEntries);

        Tensor finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()->A.elemDiv(finalB1));
    }


    @Test
    void complexDenseTest() {
        CNumber[] bEntries, expEntries;
        CTensor B, exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new CNumber[]{
                new CNumber(-0.00234, 2.452), new CNumber(15.6), new CNumber(99.2442, 9.1),
                new CNumber(100.252, 1235), new CNumber(-78.2556, -99.1441), new CNumber(0.111134, -772.4),
                new CNumber(671.455, 15.56), new CNumber(-0.00024), new CNumber(515.667, 895.52),
                new CNumber(14.515), new CNumber(100.135), new CNumber(0, 1)
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensor(bShape, bEntries);
        expEntries = new CNumber[]{
                new CNumber(aEntries[0]).div(bEntries[(0)]), new CNumber(aEntries[1]).div(bEntries[(1)]), new CNumber(aEntries[2]).div(bEntries[(2)]),
                new CNumber(aEntries[3]).div(bEntries[(3)]), new CNumber(aEntries[4]).div(bEntries[(4)]), new CNumber(aEntries[5]).div(bEntries[(5)]),
                new CNumber(aEntries[6]).div(bEntries[(6)]), new CNumber(-11154.166666666668), new CNumber(aEntries[8]).div(bEntries[(8)]),
                new CNumber(aEntries[9]).div(bEntries[(9)]), new CNumber(aEntries[10]).div(bEntries[(10)]), new CNumber(aEntries[11]).div(bEntries[(11)])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);
        assertEquals(exp, A.elemDiv(B));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new CNumber[]{
                new CNumber(-0.00234, 2.452), new CNumber(15.6), new CNumber(99.2442, 9.1),
                new CNumber(100.252, 1235), new CNumber(-78.2556, -99.1441), new CNumber(0.111134, -772.4),
                new CNumber(671.455, 15.56), new CNumber(-0.00024), new CNumber(515.667, 895.52),
                new CNumber(14.515), new CNumber(100.135), new CNumber(0, 1)
        };
        bShape = new Shape(12);
        B = new CTensor(bShape, bEntries);

        CTensor finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.elemDiv(finalB));
    }
}
