package org.flag4j.sparse_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.sparse.coo.real.RealSparseMatrixGetSet;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixSetColTests {

    @Test
    void setColTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        double[] bEntries;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.42216, 0.86886, 0.51801};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{1, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.30728, 0.13698, 0.23211, 0.05517, 0.12575};

        expShape = new Shape(5, 3);
        expEntries = new double[]{0.30728, 0.42216, 0.86886, 0.13698, 0.51801, 0.23211, 0.05517, 0.12575};
        expRowIndices = new int[]{0, 0, 0, 1, 1, 2, 3, 4};
        expColIndices = new int[]{0, 1, 2, 0, 2, 0, 0, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealSparseMatrixGetSet.setCol(a, 0, bEntries));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.86291, 0.59273, 0.14697, 0.79343, 0.0691};
        aRowIndices = new int[]{4, 5, 6, 8, 10};
        aColIndices = new int[]{14, 3, 9, 15, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.09599, 0.03342, 0.08342, 0.86195, 0.18126, 0.71121, 0.03191, 0.3479, 0.5699, 0.35584, 0.51796};

        expShape = new Shape(11, 23);
        expEntries = new double[]{0.09599, 0.03342, 0.08342, 0.86195, 0.86291, 0.18126, 0.59273, 0.71121, 0.14697, 0.03191, 0.3479, 0.79343, 0.5699, 0.35584, 0.0691, 0.51796};
        expRowIndices = new int[]{0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 10};
        expColIndices = new int[]{16, 16, 16, 16, 14, 16, 3, 16, 9, 16, 16, 15, 16, 16, 4, 16};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealSparseMatrixGetSet.setCol(a, 16, bEntries));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new double[]{0.91557, 0.99112, 0.97331, 0.46736, 0.39273, 0.9236, 0.55027, 0.96506, 0.46553};
        aRowIndices = new int[]{0, 1, 2, 3, 3, 4, 4, 4, 4};
        aColIndices = new int[]{118, 335, 419, 424, 880, 134, 358, 492, 949};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.86214, 0.01468, 0.80744, 0.38058, 0.27367};

        expShape = new Shape(5, 1000);
        expEntries = new double[]{0.91557, 0.86214, 0.99112, 0.01468, 0.97331, 0.80744, 0.46736, 0.39273, 0.38058, 0.9236, 0.55027, 0.96506, 0.46553, 0.27367};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4};
        expColIndices = new int[]{118, 999, 335, 999, 419, 999, 424, 880, 999, 134, 358, 492, 949, 999};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, RealSparseMatrixGetSet.setCol(a, 999, bEntries));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.20695, 0.08553, 0.58839, 0.42649};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 2, 4, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.70299, 0.12535, 0.51468};

        CooMatrix final0a = a;
        double[] final0b = bEntries;
        assertThrows(Exception.class, ()->RealSparseMatrixGetSet.setCol(final0a, 6, final0b));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.8715, 0.48536, 0.74835, 0.61107};
        aRowIndices = new int[]{1, 1, 2, 2};
        aColIndices = new int[]{2, 4, 2, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.18264, 0.50269, 0.62068, 0.68308, 0.25792};

        CooMatrix final1a = a;
        double[] final1b = bEntries;
        assertThrows(Exception.class, ()->RealSparseMatrixGetSet.setCol(final1a, 3, final1b));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.97644, 0.04564, 0.1204, 0.19723};
        aRowIndices = new int[]{0, 2, 2, 2};
        aColIndices = new int[]{4, 1, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.69692, 0.15703};

        CooMatrix final2a = a;
        double[] final2b = bEntries;
        assertThrows(Exception.class, ()->RealSparseMatrixGetSet.setCol(final2a, 3, final2b));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.9503, 0.0484, 0.44488, 0.29844};
        aRowIndices = new int[]{0, 2, 2, 2};
        aColIndices = new int[]{0, 1, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.36708, 0.70117, 0.73955};

        CooMatrix final3a = a;
        double[] final3b = bEntries;
        assertThrows(Exception.class, ()->RealSparseMatrixGetSet.setCol(final3a, 19, final3b));
    }


    @Test
    void setColSparseVectorTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape bShape;
        int[] bIndices;
        double[] bEntries;
        CooVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.69683, 0.7974, 0.01005};
        aRowIndices = new int[]{0, 3, 4};
        aColIndices = new int[]{1, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new double[]{0.42925, 0.95116};
        bIndices = new int[]{2, 3};
        b = new CooVector(bShape.get(0), bEntries, bIndices);

        expShape = new Shape(5, 3);
        expEntries = new double[]{0.69683, 0.42925, 0.95116, 0.01005};
        expRowIndices = new int[]{0, 2, 3, 4};
        expColIndices = new int[]{1, 0, 0, 2};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.09879, 0.44944, 0.39054, 0.51234, 0.10826};
        aRowIndices = new int[]{1, 3, 7, 8, 10};
        aColIndices = new int[]{14, 15, 16, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(11);
        bEntries = new double[]{0.42701, 0.22431, 0.48719, 0.79679};
        bIndices = new int[]{5, 6, 7, 10};
        b = new CooVector(bShape.get(0), bEntries, bIndices);

        expShape = new Shape(11, 23);
        expEntries = new double[]{0.09879, 0.44944, 0.42701, 0.22431, 0.48719, 0.51234, 0.10826, 0.79679};
        expRowIndices = new int[]{1, 3, 5, 6, 7, 8, 10, 10};
        expColIndices = new int[]{14, 15, 16, 16, 16, 3, 4, 16};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 16));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new double[]{0.548, 0.12782, 0.71044, 0.03123, 0.73197, 0.23329, 0.76449, 0.62306, 0.77283};
        aRowIndices = new int[]{0, 1, 1, 1, 2, 2, 2, 3, 4};
        aColIndices = new int[]{663, 597, 620, 926, 73, 153, 627, 66, 743};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new double[]{0.92473, 0.36888};
        bIndices = new int[]{1, 4};
        b = new CooVector(bShape.get(0), bEntries, bIndices);

        expShape = new Shape(5, 1000);
        expEntries = new double[]{0.548, 0.12782, 0.71044, 0.03123, 0.92473, 0.73197, 0.23329, 0.76449, 0.62306, 0.77283, 0.36888};
        expRowIndices = new int[]{0, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4};
        expColIndices = new int[]{663, 597, 620, 926, 999, 73, 153, 627, 66, 743, 999};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setCol(b, 999));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.38374, 0.24165, 0.20689, 0.73343};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{4, 0, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.93917};
        bIndices = new int[]{2};
        b = new CooVector(bShape.get(0), bEntries, bIndices);

        CooMatrix final0a = a;
        CooVector final0b = b;
        assertThrows(Exception.class, ()->final0a.setCol(final0b, 6));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.52077, 0.42897, 0.35701, 0.94909};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{2, 2, 0, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new double[]{0.41526, 0.41046};
        bIndices = new int[]{0, 2};
        b = new CooVector(bShape.get(0), bEntries, bIndices);

        CooMatrix final1a = a;
        CooVector final1b = b;
        assertThrows(Exception.class, ()->final1a.setCol(final1b, 3));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.89024, 0.42578, 0.66571, 0.53301};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{2, 0, 1, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2);
        bEntries = new double[]{0.55374};
        bIndices = new int[]{1};
        b = new CooVector(bShape.get(0), bEntries, bIndices);

        CooMatrix final2a = a;
        CooVector final2b = b;
        assertThrows(Exception.class, ()->final2a.setCol(final2b, 3));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.74812, 0.07704, 0.80715, 0.45783};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{0, 1, 3, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.838};
        bIndices = new int[]{0};
        b = new CooVector(bShape.get(0), bEntries, bIndices);

        CooMatrix final3a = a;
        CooVector final3b = b;
        assertThrows(Exception.class, ()->final3a.setCol(final3b, 19));
    }
}
