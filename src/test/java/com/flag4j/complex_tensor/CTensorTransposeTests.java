package com.flag4j.complex_tensor;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.Shape;
import com.flag4j.dense.CTensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CTensorTransposeTests {

    static int[] aAxes;

    static CNumber[] aEntries;
    static CNumber[] expEntries;

    static Shape aShape;
    static Shape expShape;

    static CTensor A;
    static CTensor exp;

    @BeforeEach
    void setup() {
        aEntries = new CNumber[]{
                new CNumber(1.4415, -0.0245), new CNumber(235.61, 1.45), new CNumber(0, -0.00024), 
                new CNumber(1.0), new CNumber(-85.1, 9.234), new CNumber(1.345, -781.2),
                new CNumber(0.014, -2.45),  new CNumber(-140.0),  new CNumber(0, 1.5), 
                 new CNumber(51.0, 24.56),  new CNumber(6.1, -0.03),  new CNumber(-0.00014, 1.34)};
        aShape = new Shape(3, 2, 1, 2);
        A = new CTensor(aShape, aEntries);
    }


    @Test
    void transposeTestCase() {
        // -------------------- Sub-case 1 --------------------
        expEntries = new CNumber[]{
                new CNumber(1.4415, -0.0245), new CNumber(-85.1, 9.234),  new CNumber(0, 1.5),
                new CNumber(0, -0.00024), new CNumber(0.014, -2.45),  new CNumber(6.1, -0.03),
                new CNumber(235.61, 1.45), new CNumber(1.345, -781.2),  new CNumber(51.0, 24.56),
                new CNumber(1.0),  new CNumber(-140.0),  new CNumber(-0.00014, 1.34)};
        expShape = new Shape(2, 2, 1, 3);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.transpose());

        // -------------------- Sub-case 2 --------------------
        aAxes = new int[]{0, 2, 3, 1};
        expEntries = new CNumber[]{
                new CNumber(1.4415, -0.0245), new CNumber(0, -0.00024), new CNumber(235.61, 1.45),
                new CNumber(1.0), new CNumber(-85.1, 9.234), new CNumber(0.014, -2.45),
                new CNumber(1.345, -781.2),  new CNumber(-140.0),  new CNumber(0, 1.5),
                new CNumber(6.1, -0.03),  new CNumber(51.0, 24.56),  new CNumber(-0.00014, 1.34)
        };
        expShape = new Shape(3, 1, 2, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.transpose(aAxes));

        // -------------------- Sub-case 3 --------------------
        aAxes = new int[]{3, 2, 1, 0};
        expEntries = new CNumber[]{
                new CNumber(1.4415, -0.0245), new CNumber(-85.1, 9.234),  new CNumber(0, 1.5),
                new CNumber(0, -0.00024), new CNumber(0.014, -2.45),  new CNumber(6.1, -0.03),
                new CNumber(235.61, 1.45), new CNumber(1.345, -781.2),  new CNumber(51.0, 24.56),
                new CNumber(1.0),  new CNumber(-140.0),  new CNumber(-0.00014, 1.34)
        };
        expShape = new Shape(2, 1, 2, 3);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.transpose(aAxes));

        // -------------------- Sub-case 4 --------------------
        expEntries = new CNumber[]{
                new CNumber(1.4415, -0.0245), new CNumber(0, -0.00024), new CNumber(235.61, 1.45),
                new CNumber(1.0), new CNumber(-85.1, 9.234), new CNumber(0.014, -2.45),
                new CNumber(1.345, -781.2),  new CNumber(-140.0),  new CNumber(0, 1.5),
                new CNumber(6.1, -0.03),  new CNumber(51.0, 24.56),  new CNumber(-0.00014, 1.34)
        };
        expShape = new Shape(3, 2, 1, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.transpose(3, 1));

        // -------------------- Sub-case 5 --------------------
        aAxes = new int[]{3, 2, 1, 2};
        assertThrows(IllegalArgumentException.class, ()->A.transpose(aAxes));

        // -------------------- Sub-case 6 --------------------
        aAxes = new int[]{3, 2, 1};
        assertThrows(IllegalArgumentException.class, ()->A.transpose(aAxes));

        // -------------------- Sub-case 7 --------------------
        aAxes = new int[]{0, 1, 3, 2, 4};
        assertThrows(IllegalArgumentException.class, ()->A.transpose(aAxes));

        // -------------------- Sub-case 8 --------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.transpose(-1, 0));

        // -------------------- Sub-case 9 --------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.transpose(1, 6));
    }


    @Test
    void hermTransposeTestCase() {
        // -------------------- Sub-case 1 --------------------
        expEntries = new CNumber[]{
                new CNumber(1.4415, -0.0245), new CNumber(-85.1, 9.234),  new CNumber(0, 1.5),
                new CNumber(0, -0.00024), new CNumber(0.014, -2.45),  new CNumber(6.1, -0.03),
                new CNumber(235.61, 1.45), new CNumber(1.345, -781.2),  new CNumber(51.0, 24.56),
                new CNumber(1.0),  new CNumber(-140.0),  new CNumber(-0.00014, 1.34)};
        expShape = new Shape(2, 2, 1, 3);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp.conj(), A.hermTranspose());

        // -------------------- Sub-case 2 --------------------
        aAxes = new int[]{0, 2, 3, 1};
        expEntries = new CNumber[]{
                new CNumber(1.4415, -0.0245), new CNumber(0, -0.00024), new CNumber(235.61, 1.45),
                new CNumber(1.0), new CNumber(-85.1, 9.234), new CNumber(0.014, -2.45),
                new CNumber(1.345, -781.2),  new CNumber(-140.0),  new CNumber(0, 1.5),
                new CNumber(6.1, -0.03),  new CNumber(51.0, 24.56),  new CNumber(-0.00014, 1.34)
        };
        expShape = new Shape(3, 1, 2, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp.conj(), A.hermTranspose(aAxes));

        // -------------------- Sub-case 3 --------------------
        aAxes = new int[]{3, 2, 1, 0};
        expEntries = new CNumber[]{
                new CNumber(1.4415, -0.0245), new CNumber(-85.1, 9.234),  new CNumber(0, 1.5),
                new CNumber(0, -0.00024), new CNumber(0.014, -2.45),  new CNumber(6.1, -0.03),
                new CNumber(235.61, 1.45), new CNumber(1.345, -781.2),  new CNumber(51.0, 24.56),
                new CNumber(1.0),  new CNumber(-140.0),  new CNumber(-0.00014, 1.34)
        };
        expShape = new Shape(2, 1, 2, 3);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp.conj(), A.hermTranspose(aAxes));

        // -------------------- Sub-case 4 --------------------
        expEntries = new CNumber[]{
                new CNumber(1.4415, -0.0245), new CNumber(0, -0.00024), new CNumber(235.61, 1.45),
                new CNumber(1.0), new CNumber(-85.1, 9.234), new CNumber(0.014, -2.45),
                new CNumber(1.345, -781.2),  new CNumber(-140.0),  new CNumber(0, 1.5),
                new CNumber(6.1, -0.03),  new CNumber(51.0, 24.56),  new CNumber(-0.00014, 1.34)
        };
        expShape = new Shape(3, 2, 1, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp.conj(), A.hermTranspose(3, 1));

        // -------------------- Sub-case 5 --------------------
        aAxes = new int[]{3, 2, 1, 2};
        assertThrows(IllegalArgumentException.class, ()->A.hermTranspose(aAxes));

        // -------------------- Sub-case 6 --------------------
        aAxes = new int[]{3, 2, 1};
        assertThrows(IllegalArgumentException.class, ()->A.hermTranspose(aAxes));

        // -------------------- Sub-case 7 --------------------
        aAxes = new int[]{0, 1, 3, 2, 4};
        assertThrows(IllegalArgumentException.class, ()->A.hermTranspose(aAxes));

        // -------------------- Sub-case 8 --------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.hermTranspose(-1, 0));

        // -------------------- Sub-case 9 --------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.hermTranspose(1, 6));
    }
}
