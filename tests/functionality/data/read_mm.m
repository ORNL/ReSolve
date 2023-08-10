function Mat = read_mtx(filename)
    % open file
    fID = fopen(filename,'r');
    M = fscanf(fID, '%f');
    % reshape M vector into Nx3 matrix
    M = reshape(M, [3, length(M)/3])';
    % assemble final matrix
    Mat = zeros(M(end,1),M(end,1));
    for ii = 1:size(M,1)
        Mat(M(ii,1),M(ii,2)) = M(ii,3);
    end
end
