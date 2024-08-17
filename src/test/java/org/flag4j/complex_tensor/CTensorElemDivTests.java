package org.flag4j.complex_tensor;


import org.flag4j.arrays_old.dense.CTensorOld;
import org.flag4j.arrays_old.dense.TensorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CTensorElemDivTests {
    static CNumber[] aEntries ,expEntries;
    static CTensorOld A, exp;
    static Shape aShape, bShape, expShape;

    @BeforeEach
    void setup() {
        aEntries = new CNumber[]{
                new CNumber(1.4415, -0.0245), new CNumber(235.61, 1.45), new CNumber(0, -0.00024),
                new CNumber(1.0), new CNumber(-85.1, 9.234), new CNumber(1.345, -781.2),
                new CNumber(0.014, -2.45),  new CNumber(-140.0),  new CNumber(0, 1.5),
                new CNumber(51.0, 24.56),  new CNumber(6.1, -0.03),  new CNumber(-0.00014, 1.34),};
        aShape = new Shape(2, 3, 2);
        A = new CTensorOld(aShape, aEntries);
    }

    @Test
    void realDenseTestCase() {
        double[] bEntries;
        TensorOld B;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2);
        B = new TensorOld(bShape, bEntries);
        expEntries = new CNumber[]{
                aEntries[0].div(bEntries[0]), aEntries[1].div(bEntries[1]), aEntries[2].div(bEntries[2]),
                aEntries[3].div(bEntries[3]), aEntries[4].div(bEntries[4]), aEntries[5].div(bEntries[5]),
                aEntries[6].div(bEntries[6]), aEntries[7].div(bEntries[7]), aEntries[8].div(bEntries[8]),
                aEntries[9].div(bEntries[9]), aEntries[10].div(bEntries[10]), aEntries[11].div(bEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensorOld(expShape, expEntries);

        assertEquals(exp, A.elemDiv(B));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2, 1);
        B = new TensorOld(bShape, bEntries);

        TensorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.elemDiv(finalB));

        // ----------------------- Sub-case 3 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0, 1.4, 5
        };
        bShape = new Shape(7, 2);
        B = new TensorOld(bShape, bEntries);

        TensorOld finalB1 = B;
        assertThrows(LinearAlgebraException.class, ()->A.elemDiv(finalB1));
    }


    @Test
    void complexDenseTestCase() {
        CNumber[] bEntries;
        CTensorOld B;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new CNumber[]{
                new CNumber(-0.00234, 2.452), new CNumber(15.6), new CNumber(99.2442, 9.1),
                new CNumber(100.252, 1235), new CNumber(-78.2556, -99.1441), new CNumber(0.111134, -772.4),
                new CNumber(671.455, 15.56), new CNumber(-0.00024), new CNumber(515.667, 895.52),
                new CNumber(14.515), new CNumber(100.135), new CNumber(0, 1)
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensorOld(bShape, bEntries);
        expEntries = new CNumber[]{
                aEntries[0].div(bEntries[0]), aEntries[1].div(bEntries[1]), aEntries[2].div(bEntries[2]),
                aEntries[3].div(bEntries[3]), aEntries[4].div(bEntries[4]), aEntries[5].div(bEntries[5]),
                aEntries[6].div(bEntries[6]), aEntries[7].div(bEntries[7]), aEntries[8].div(bEntries[8]),
                aEntries[9].div(bEntries[9]), aEntries[10].div(bEntries[10]), aEntries[11].div(bEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensorOld(expShape, expEntries);

        assertEquals(exp, A.elemDiv(B));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new CNumber[]{
                new CNumber(-0.00234, 2.452), new CNumber(15.6), new CNumber(99.2442, 9.1),
                new CNumber(100.252, 1235), new CNumber(-78.2556, -99.1441), new CNumber(0.111134, -772.4),
                new CNumber(671.455, 15.56), new CNumber(-0.00024), new CNumber(515.667, 895.52),
                new CNumber(14.515), new CNumber(100.135), new CNumber(0, 1)
        };
        bShape = new Shape(12);
        B = new CTensorOld(bShape, bEntries);

        CTensorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.elemDiv(finalB));
    }
}
