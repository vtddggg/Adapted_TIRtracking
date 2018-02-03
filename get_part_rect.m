function [ annotations ] = get_part_rect(box,npart)
    width = box(3);
    height = box(4);
    if npart ==2
        annotations = cell(1,2);
        if height>width
        annotations{1}=[box(1) box(2)  width  height/2];
        annotations{2}=[box(1) box(2)+height/2  width  height/2];

        else
        annotations{1}=[box(1) box(2) width/2 height];
        annotations{2}=[box(1)+width/2 box(2) width/2 height];
        end
    elseif npart >2
        multiplier = sqrt(npart);
        if multiplier ~= fix(multiplier) 
            error('Invalid npart number!') ;
        end
        annotations = cell(multiplier,multiplier);
        for i = 1:multiplier
            for j = 1:multiplier
               annotations{i,j} =  [box(1)+(j-1)*width/multiplier box(2)+(i-1)*height/multiplier width/multiplier height/multiplier];
            end
        end
        
    else
        error('Invalid npart number!') ;
    end

end

